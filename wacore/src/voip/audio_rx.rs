//! Receive-side counters for a 1:1 call's audio path, used to see where inbound packets die.
//!
//! Counters are plain integers bumped inline by the engine; only the summary line is gated on the
//! `tracing` feature. Nothing here records payload bytes or key material.

use super::engine::Millis;

/// Minimum spacing of the periodic summary on the engine's `now` clock.
const SUMMARY_INTERVAL_MS: Millis = 2000;
/// Cap on distinct payload types / SSRCs tracked for the "first seen" summary.
const MAX_TRACKED: usize = 8;

#[derive(Debug, Default)]
pub(crate) struct AudioRxStats {
    pub total: u64,
    pub header_parse_failed: u64,
    pub no_media: u64,
    pub video_pt: u64,
    pub audio_pt_rejected: u64,
    pub last_rejected_pt: Option<u8>,
    pub unprotect_failed: u64,
    pub unprotected: u64,
    pub sframe_decrypted: u64,
    pub sframe_passthrough: u64,
    pub foreign_audio: u64,
    pub encoded_audio: u64,
    pub mlow_decoded: u64,
    pub mlow_samples: u64,
    pub no_pcm: u64,
    /// Audio packets split off a video, FEC or app-data packet and handed to the decode path.
    pub piggybacked: u64,
    /// Piggybacked audio split off an authenticated packet that could not be decoded: unreadable
    /// header, payload type outside the active profile, or no payload.
    pub piggyback_failed: u64,
    /// Relay packets that were neither STUN, RTCP nor RTP, plus malformed group envelopes.
    pub demux_other: u64,
    pub forwarding_unwrap_failed: u64,
    last_pt: Option<u8>,
    last_ssrc: Option<u32>,
    seen_pts: Vec<u8>,
    seen_ssrcs: Vec<u32>,
    new_pt_or_ssrc: bool,
    last_summary: Option<Millis>,
}

impl AudioRxStats {
    /// Record the payload type and SSRC of a non-video packet, flagging the first sighting of each.
    pub(crate) fn observe_audio_header(&mut self, payload_type: u8, ssrc: u32) {
        self.last_pt = Some(payload_type);
        self.last_ssrc = Some(ssrc);
        if !self.seen_pts.contains(&payload_type) && self.seen_pts.len() < MAX_TRACKED {
            self.seen_pts.push(payload_type);
            self.new_pt_or_ssrc = true;
        }
        if !self.seen_ssrcs.contains(&ssrc) && self.seen_ssrcs.len() < MAX_TRACKED {
            self.seen_ssrcs.push(ssrc);
            self.new_pt_or_ssrc = true;
        }
    }

    /// Whether a summary should be emitted now; consumes the "first seen" flag and restarts the
    /// interval when it says yes. Silent until something has been counted.
    pub(crate) fn summary_due(&mut self, now: Millis) -> bool {
        if self.total == 0 && self.demux_other == 0 && self.forwarding_unwrap_failed == 0 {
            return false;
        }
        let periodic = self
            .last_summary
            .is_none_or(|last| now.saturating_sub(last) >= SUMMARY_INTERVAL_MS);
        if !periodic && !self.new_pt_or_ssrc {
            return false;
        }
        self.new_pt_or_ssrc = false;
        self.last_summary = Some(now);
        true
    }

    pub(crate) fn emit(&self, call_id: &str) {
        log::debug!(
            target: "wacore::voip::audio_rx",
            "voip audio rx call_id={call_id} total={} header_parse_failed={} no_media={} video_pt={} \
             audio_pt_rejected={} last_rejected_pt={:?} unprotect_failed={} unprotected={} \
             sframe_decrypted={} sframe_passthrough={} foreign_audio={} encoded_audio={} \
             mlow_decoded={} mlow_samples={} no_pcm={} piggybacked={} piggyback_failed={} \
             demux_other={} forwarding_unwrap_failed={} \
             last_pt={:?} last_ssrc={:?}",
            self.total,
            self.header_parse_failed,
            self.no_media,
            self.video_pt,
            self.audio_pt_rejected,
            self.last_rejected_pt,
            self.unprotect_failed,
            self.unprotected,
            self.sframe_decrypted,
            self.sframe_passthrough,
            self.foreign_audio,
            self.encoded_audio,
            self.mlow_decoded,
            self.mlow_samples,
            self.no_pcm,
            self.piggybacked,
            self.piggyback_failed,
            self.demux_other,
            self.forwarding_unwrap_failed,
            self.last_pt,
            self.last_ssrc,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_until_a_packet_is_counted() {
        let mut s = AudioRxStats::default();
        assert!(!s.summary_due(10_000));
    }

    #[test]
    fn first_sighting_emits_then_interval_gates() {
        let mut s = AudioRxStats {
            total: 1,
            ..Default::default()
        };
        s.observe_audio_header(120, 7);
        assert!(s.summary_due(100));
        s.total = 2;
        s.observe_audio_header(120, 7);
        assert!(!s.summary_due(1_000), "same pt/ssrc inside 2000 ms");
        assert!(!s.summary_due(2_099));
        assert!(s.summary_due(2_100));
        assert!(!s.summary_due(2_101));
    }

    #[test]
    fn summary_due_on_first_sighting_not_within_2000ms_then_again() {
        let mut s = AudioRxStats {
            total: 1,
            ..Default::default()
        };
        s.observe_audio_header(120, 7);
        assert!(s.summary_due(10_000));
        assert!(!s.summary_due(11_999));
        assert!(s.summary_due(12_000));
    }

    #[test]
    fn new_pt_or_ssrc_emits_immediately() {
        let mut s = AudioRxStats {
            total: 1,
            ..Default::default()
        };
        s.observe_audio_header(120, 7);
        assert!(s.summary_due(0));
        s.observe_audio_header(121, 7);
        assert!(s.summary_due(5), "new pt");
        s.observe_audio_header(121, 8);
        assert!(s.summary_due(6), "new ssrc");
        s.observe_audio_header(121, 8);
        assert!(!s.summary_due(7));
    }

    #[test]
    fn tracked_sets_are_capped() {
        let mut s = AudioRxStats {
            total: 1,
            ..Default::default()
        };
        for pt in 0..20u8 {
            s.observe_audio_header(pt, 1);
        }
        assert_eq!(s.seen_pts.len(), MAX_TRACKED);
        assert!(s.summary_due(0));
        s.observe_audio_header(99, 1);
        assert!(!s.summary_due(1), "past the cap, a new pt is not novel");
    }
}
