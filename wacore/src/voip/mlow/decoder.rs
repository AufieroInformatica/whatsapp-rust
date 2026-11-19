//! MLow top-level decoder: RED strip -> TOC routing -> active-frame decode (3 chained 20 ms internal
//! frames: LSF -> pulses -> pitch/gains -> CELP synthesis) -> 60 ms PCM. The synthesis
//! (`smpl_celpdec`) runs the excitation in the codec's float domain (gen_noise + LPC synthesis). The
//! cross-frame predictor and synthesis history persist across calls because the stream is
//! continuous.

use super::multiframe::{is_multiframe, split_multiframe};
use super::rangecoder::RangeDecoder;
use super::red::depack_split_red;
use super::smpl_cc_tables::load_cc_tables;
use super::smpl_celpdec::CelpDecParams;
use super::smpl_decode::{decode_smpl_lsf, load_smpl_tables};
use super::smpl_gains::decode_smpl_gains;
use super::smpl_mem::load_smpl_mem;
use super::smpl_pitch::decode_smpl_pitch;
use super::smpl_pulse::decode_smpl_pulses;
use super::smpl_synth::{
    SMPL_INTF_LEN, SmplDecoderState, load_smpl_synth_tables, smpl_reconstruct_nlsf,
};
use super::toc::parse_mlow_toc;

const OPUS_FRAME_SAMPS: usize = 960; // 60 ms @ 16 kHz

/// The longest audio one packet may carry (120 ms @ 16 kHz): the longest duration a TOC can declare
/// and the ceiling the shipped client applies when it totals a multiframe envelope. A hard bound:
/// without it an envelope of eighteen sub-frames expands a few hundred bytes into thousands of
/// samples that the jitter buffer cannot tell from real audio.
const MAX_PACKET_SAMPS: usize = 1920;

/// Stateful pure-Rust MLow decoder. Decodes one RTP payload (a bare MLow frame, a multiframe
/// envelope, or a SplitRed packet when redundancy was negotiated) into PCM at 16 kHz: 960 samples
/// for a single 60 ms frame, one such slot per sub-frame for an envelope.
pub struct MlowDecoder {
    state: SmplDecoderState,
    redundancy: i32,
    /// Sticky: set whenever the inner range decoder raised its error flag during any decode. That flag
    /// reflects a degenerate decode table, not arbitrary frame corruption (over-reads return zero
    /// silently), so it does not detect a tampered payload. Read via `had_error`. Diagnostic only,
    /// never gates output.
    had_error: bool,
    /// Count of inbound frames dropped because they fall outside this decoder's single operating point
    /// (16kHz wideband, low_rate=0, 60ms). Such a frame would desync the range coder if decoded, so it
    /// is dropped (treated as a lost frame). The count drives a once + every-100th `warn` (which names
    /// the offending dimension) so a live capture reveals whether real peers emit these (decides the
    /// follow-ups).
    dropped_unsupported: u32,
    /// Malformed multiframe envelopes concealed so far; drives a once + every-100th `warn`.
    malformed: u32,
    /// Whether the most recent `decode` call unpacked a multiframe envelope.
    last_was_multiframe: bool,
}

impl Default for MlowDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl MlowDecoder {
    pub fn new() -> Self {
        MlowDecoder {
            state: SmplDecoderState::default(),
            redundancy: 0,
            had_error: false,
            dropped_unsupported: 0,
            malformed: 0,
            last_was_multiframe: false,
        }
    }

    /// Whether any decode since construction (or `reset`) raised the range decoder's error flag (a
    /// degenerate decode table). It does not flag a corrupted payload, which the decoder absorbs.
    /// Diagnostic only; consumed by the regression suites, so it is gated to test builds.
    #[cfg(test)]
    pub(crate) fn had_error(&self) -> bool {
        self.had_error
    }

    /// Whether the most recent `decode` call unpacked a multiframe envelope (also one carried inside
    /// a RED wrapper), so a caller can count them.
    pub fn last_was_multiframe(&self) -> bool {
        self.last_was_multiframe
    }

    /// Set the negotiated RED redundancy level (0 = bare frames, the common case).
    pub fn set_redundancy(&mut self, n: i32) {
        self.redundancy = n;
    }

    /// Clear the cross-frame state (call at a stream discontinuity).
    pub fn reset(&mut self) {
        self.state = SmplDecoderState::default();
        self.had_error = false;
    }

    /// Decode one RTP MLow payload into PCM, float in [-1, 1]. A single frame yields a 60 ms
    /// (960-sample) slot; a multiframe envelope yields the concatenation of its sub-frames.
    pub fn decode(&mut self, payload: &[u8]) -> Vec<f32> {
        self.last_was_multiframe = false;
        if payload.is_empty() {
            return vec![0.0; OPUS_FRAME_SAMPS];
        }
        if self.redundancy > 0 {
            return match depack_split_red(payload) {
                // the main (current) frame is last; its slice borrows `payload`, not `self`, so it
                // can drive the decode directly (no copy). Through `decode_inner`, not
                // `decode_frame`: the frame inside a RED wrapper can itself be a multiframe
                // envelope, which `decode_frame` would read as a SID.
                Ok(frames) => match frames.last() {
                    Some(main) => self.decode_inner(main.data),
                    None => self.decode_frame(&[]),
                },
                Err(e) => {
                    log::warn!("mlow RED depacketization failed: {e:?}");
                    vec![0.0; OPUS_FRAME_SAMPS]
                }
            };
        }
        self.decode_inner(payload)
    }

    /// Route one bare MLow payload: multiframe envelope, or a single frame. The envelope is checked
    /// before the TOC is read, because its indicator byte has bit 7 set and would read as a SID.
    fn decode_inner(&mut self, payload: &[u8]) -> Vec<f32> {
        if is_multiframe(payload) {
            return self.decode_multiframe(payload);
        }
        self.decode_frame(payload)
    }

    /// Decode every sub-frame of an envelope, in transmission order, into one contiguous packet.
    /// The envelope's RTP timestamp belongs to the LAST sub-frame and the earlier ones run backwards
    /// from it, so array order is time order.
    fn decode_multiframe(&mut self, payload: &[u8]) -> Vec<f32> {
        let frames = match split_multiframe(payload) {
            Ok(frames) => frames,
            Err(e) => {
                self.note_malformed(&format!("failed to parse ({e})"), payload[0]);
                return vec![0.0; OPUS_FRAME_SAMPS];
            }
        };
        // Refuse before decoding anything: half-decoding an over-long packet would advance the
        // predictor by frames the peer never meant as one packet.
        let declared: usize = frames
            .iter()
            .map(|f| output_samps(f.first().copied().unwrap_or(0)))
            .sum();
        if declared > MAX_PACKET_SAMPS {
            self.note_malformed(
                &format!("declares {declared} samples, past the {MAX_PACKET_SAMPS} ceiling"),
                payload[0],
            );
            return vec![0.0; OPUS_FRAME_SAMPS];
        }
        self.last_was_multiframe = true;
        let mut out = Vec::with_capacity(declared);
        for frame in frames {
            out.extend_from_slice(&self.decode_frame(frame));
        }
        out
    }

    fn note_malformed(&mut self, what: &str, indicator: u8) {
        self.malformed += 1;
        if self.malformed == 1 || self.malformed.is_multiple_of(100) {
            log::warn!(
                "mlow: multiframe packet #{} {what} (indicator 0x{indicator:02x}); concealed as one lost frame",
                self.malformed
            );
        }
    }

    fn decode_frame(&mut self, frame: &[u8]) -> Vec<f32> {
        if frame.is_empty() {
            return vec![0.0; OPUS_FRAME_SAMPS];
        }
        let toc = parse_mlow_toc(frame[0]);
        if toc.std_opus {
            let out_len = (16000 / 1000 * toc.frame_ms) as usize;
            log::debug!(
                "mlow: standard-opus TOC 0x{:02x} -> {out_len} samples silence",
                frame[0]
            );
            return vec![0.0; out_len];
        }
        // Inactive / SID (DTX/CNG) frames carry no decodable voice and are silenced without opening the
        // range coder, so their geometry can never desync. Handle them before the operating-point guard:
        // otherwise an inactive off-point frame (e.g. the 10ms startup silence a real peer emits before
        // speech) would trip the loud "dropped" canary instead of being the benign silence it is. A full
        // 60ms slot keeps the playout cadence regardless of the frame's nominal duration.
        if toc.sid || !toc.active {
            log::debug!("mlow: DTX/SID TOC 0x{:02x} -> 60ms silence", frame[0]);
            return vec![0.0; OPUS_FRAME_SAMPS];
        }
        // Operating-point guard for active frames: an active frame at a different internal rate, the
        // low_rate=1 2x160 geometry, or a non-60ms duration would desync the range coder, since
        // decode_active_frame always runs the 3x20ms / 60ms geometry and would consume the payload with
        // the wrong symbol count (garbage plus a poisoned cross-frame predictor that propagates to later
        // packets). Drop it as a lost frame so the predictor holds its last good values. flag2 is the
        // smpl TOC's low_rate bit; the warn names the offending dimension so a live capture shows whether
        // real peers ever emit active out-of-point frames in 1:1 calls.
        let off_point = if toc.sample_rate != 16000 {
            Some(("rate", i64::from(toc.sample_rate / 1000)))
        } else if toc.flag2 {
            Some(("low_rate", 1))
        } else if toc.frame_ms != 60 {
            Some(("frame_ms", i64::from(toc.frame_ms)))
        } else {
            None
        };
        if let Some((dim, val)) = off_point {
            self.dropped_unsupported += 1;
            if self.dropped_unsupported == 1 || self.dropped_unsupported.is_multiple_of(100) {
                log::warn!(
                    "mlow: dropping out-of-operating-point frame #{} ({dim}={val}, TOC 0x{:02x}); \
                     the 1:1 decoder is 16kHz / low_rate=0 / 60ms only",
                    self.dropped_unsupported,
                    frame[0]
                );
            }
            return vec![0.0; OPUS_FRAME_SAMPS];
        }
        self.decode_active_frame(frame, OPUS_FRAME_SAMPS)
    }

    fn decode_active_frame(&mut self, frame: &[u8], out_len: usize) -> Vec<f32> {
        let config = (frame[0] >> 2) as usize & 1;
        let tbl = load_smpl_tables();
        let synth_t = load_smpl_synth_tables();
        let mem = load_smpl_mem();
        let cc = load_cc_tables();
        let mut dec = RangeDecoder::new(&frame[1..]);

        // The low_rate bit of the smpl TOC (this capture is low_rate==0; the synth gates on it).
        let low_rate = (frame[0] >> 2) & 1 != 0;

        let mut out: Vec<f32> = Vec::with_capacity(3 * SMPL_INTF_LEN);
        // Collect the per-40-block lags (8 per frame, 24 per packet) and the average normalized
        // bitrate for the per-packet harmonic postfilter.
        let mut packet_lags: Vec<f32> = Vec::with_capacity(3 * 8);
        let mut avg_norm_br = 0.0f32;
        for f in 0..3 {
            let lsf = decode_smpl_lsf(&mut dec, tbl, &mut self.state.lstate, config, f);
            let pulses = decode_smpl_pulses(
                &mut dec,
                cc,
                SMPL_INTF_LEN as i32,
                4,
                1,
                config as i32,
                lsf.stage1,
            );
            let voiced = lsf.stage1 == 1;
            let mut params = CelpDecParams {
                voiced,
                sf_pulses: pulses.subfr,
                fcbg_idx: [0; 4],
                nrgres_dbq_q14: [0; 4],
                acbg_idx: [0; 4],
                block_lags: [0.0; 8],
                total_pulses: pulses.subfr.iter().sum(),
            };
            if voiced {
                let pr = decode_smpl_pitch(
                    &mut dec,
                    mem,
                    cc,
                    &mut self.state.lstate,
                    SMPL_INTF_LEN as i32,
                    4,
                    config as i32,
                    pulses.subfr,
                );
                // lag = laginds*0.5 + SMPL_MIN_PITCH_LAG, clamped; one per 40-block, 8 per frame.
                for b in 0..8 {
                    params.block_lags[b] =
                        ((pr.block_lags[b] as f64 * 0.5 + 32.0).min(320.0)) as f32;
                }
                for sf in 0..4 {
                    params.acbg_idx[sf] = pr.gain_idx[sf];
                    // The voiced FCB gain index is decoded in the pitch block (filt_idx).
                    params.fcbg_idx[sf] = pr.filt_idx[sf].max(0);
                }
            } else {
                let g = decode_smpl_gains(&mut dec, cc, 4, pulses.subfr);
                // The unvoiced gains decode yields gain_q (the nrgres_dbq_Q14 field) and nrg_res (the
                // fcbg_idx field).
                params.nrgres_dbq_q14 = g.gain_q;
                params.fcbg_idx = g.nrg_res;
            }
            packet_lags.extend_from_slice(&params.block_lags);
            avg_norm_br += super::smpl_gennoise::smpl_get_normalized_bitrate(
                params.total_pulses,
                SMPL_INTF_LEN as i32,
            );

            let nlsf = smpl_reconstruct_nlsf(
                synth_t,
                lsf.stage1 as usize,
                config,
                lsf.grid as usize,
                &lsf.stage2,
                &self.state.prev_nlsf,
            );
            let mut sig = [0f32; SMPL_INTF_LEN];
            self.state.celp.synth_frame(
                &nlsf,
                lsf.extra as usize,
                &pulses.pulses,
                &params,
                low_rate,
                SMPL_INTF_LEN as i32,
                &mut sig,
            );
            self.state.prev_nlsf = nlsf;
            out.extend_from_slice(&sig);
        }

        // Per-packet harmonic postfilter (the codec's final pitch comb + 48-sample group delay), run
        // once over the whole packet with the 24 per-40-block lags and the average normalized bitrate.
        let plen = out.len();
        super::smpl_harm_postfilter::smpl_harm_postfilter(
            &mut self.state.harm,
            &mut out,
            plen,
            &packet_lags,
            packet_lags.len(),
            avg_norm_br / 3.0,
        );

        // The C-domain synthesis output is already float in [-1, 1]; clamp in place.
        for v in &mut out {
            *v = v.clamp(-1.0, 1.0);
        }
        if out_len > 0 && out_len != out.len() {
            out.resize(out_len, 0.0);
        }
        if dec.err != 0 {
            // Sticky flag for `had_error`; does not alter `out` (the frame still plays).
            self.had_error = true;
            log::warn!("mlow: range decoder raised its error flag after active-frame decode");
        }
        log::debug!(
            "mlow: active frame decoded -> {} samples (config={config})",
            out.len()
        );
        out
    }
}

/// Samples `decode_frame` emits for a frame with this TOC byte: its own length for the standard-Opus
/// escape, otherwise one 60 ms slot (active frames at that duration, and every silenced or dropped
/// frame).
fn output_samps(toc_byte: u8) -> usize {
    let toc = parse_mlow_toc(toc_byte);
    if toc.std_opus {
        (16000 / 1000 * toc.frame_ms) as usize
    } else {
        OPUS_FRAME_SAMPS
    }
}

/// Per-subframe param snapshot for the param-decode-match (T1) test.
#[cfg(test)]
pub(crate) struct DiagParam {
    pub(crate) packet: usize,
    pub(crate) frame: usize,
    pub(crate) sf: usize,
    pub(crate) voiced: bool,
    /// The `gain_q` value, i.e. the `nrgres_dbq_Q14` field.
    pub(crate) nrgres_dbq_q14: i32,
    /// The per-subframe `nrg_res` / voiced `filt_idx` symbol, i.e. the `fcbg_idx` field.
    pub(crate) fcbg_idx: i32,
}

/// Re-run the active-frame decode over the capture and capture per-subframe unvoiced params, keyed
/// by (packet, frame, sf), to compare against the reference dump (see testdata/PROVENANCE.md).
#[cfg(test)]
pub(crate) fn diag_decode_params() -> Vec<DiagParam> {
    let frames: Vec<String> =
        serde_json::from_str(include_str!("testdata/inbound_capture_frames.json")).unwrap();
    let tbl = load_smpl_tables();
    let mem = load_smpl_mem();
    let cc = load_cc_tables();
    let mut lstate = super::smpl_decode::SmplLsfState::default();
    let mut out = Vec::new();
    for (packet, hex_frame) in frames.iter().enumerate() {
        let frame = hex::decode(hex_frame).unwrap();
        if frame.is_empty() {
            continue;
        }
        let toc = parse_mlow_toc(frame[0]);
        if toc.std_opus || toc.sid || !toc.active {
            continue;
        }
        let config = (frame[0] >> 2) as usize & 1;
        let mut dec = RangeDecoder::new(&frame[1..]);
        for f in 0..3 {
            let lsf = decode_smpl_lsf(&mut dec, tbl, &mut lstate, config, f);
            let pulses = decode_smpl_pulses(
                &mut dec,
                cc,
                SMPL_INTF_LEN as i32,
                4,
                1,
                config as i32,
                lsf.stage1,
            );
            if lsf.stage1 == 1 {
                let pr = decode_smpl_pitch(
                    &mut dec,
                    mem,
                    cc,
                    &mut lstate,
                    SMPL_INTF_LEN as i32,
                    4,
                    config as i32,
                    pulses.subfr,
                );
                for sf in 0..4 {
                    out.push(DiagParam {
                        packet,
                        frame: f,
                        sf,
                        voiced: true,
                        nrgres_dbq_q14: pr.gain_idx[sf],
                        fcbg_idx: pr.filt_idx[sf],
                    });
                }
            } else {
                let g = decode_smpl_gains(&mut dec, cc, 4, pulses.subfr);
                for sf in 0..4 {
                    out.push(DiagParam {
                        packet,
                        frame: f,
                        sf,
                        voiced: false,
                        nrgres_dbq_q14: g.gain_q[sf],
                        fcbg_idx: g.nrg_res[sf],
                    });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // End-to-end: decode the whole capture and compare against the reference output
    // (`ref_usesmpl_expected.raw`; see testdata/PROVENANCE.md).
    //
    // An earlier target (`e2e_vectors.json`) was proven wrong: it used the int16-domain `*nrgres`
    // excitation with no shaped noise (a tail-off bug) and correlates ~0 with the true codec. With
    // the per-block voiced ACB/LTP lags, the HP postfilter, and the harmonic postfilter (which emits
    // the SMPL_TOT_POSTFILT_DELAY = 48-sample group delay) all in place, the decode now aligns
    // sample-for-sample at lag 0.
    #[test]
    fn e2e_decode_matches_usesmpl() {
        let frames: Vec<String> =
            serde_json::from_str(include_str!("testdata/inbound_capture_frames.json"))
                .expect("inbound_capture_frames.json");
        let refp: Vec<f32> = include_bytes!("testdata/ref_usesmpl_expected.raw")
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        let mut dec = MlowDecoder::new();
        let mut out: Vec<f32> = Vec::new();
        for hex_frame in &frames {
            let frame = hex::decode(hex_frame).unwrap();
            out.extend_from_slice(&dec.decode(&frame));
        }
        assert_eq!(out.len(), refp.len(), "decode length vs reference");

        // Aligned at lag 0 now (the harmonic postfilter emits the 48-sample group delay).
        const LAG: usize = 0;
        let n = refp.len() - LAG;
        let (r, o) = (&refp[LAG..LAG + n], &out[..n]);
        let mr: f64 = r.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let mo: f64 = o.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let (mut sxy, mut sxx, mut syy) = (0f64, 0f64, 0f64);
        for i in 0..n {
            let dr = r[i] as f64 - mr;
            let dz = o[i] as f64 - mo;
            sxy += dr * dz;
            sxx += dr * dr;
            syy += dz * dz;
        }
        let corr = sxy / (sxx * syy).sqrt();
        assert!(corr > 0.95, "lag-0 corr {corr:.4} vs reference");
    }

    // R2 (fuzz no-panic): the decoder is fed adversarial inputs and must neither panic nor over-emit.
    // Corpus: a deterministic LCG of random byte vectors, plus every capture frame with each single
    // byte flipped and each prefix truncation. The contract is purely structural (no panic, bounded
    // output); the range decoder absorbs corruption by returning zero, so `had_error` is not asserted.
    //
    // Output length is data-driven by the TOC: `sample_rate/1000 * frame_ms`, where the TOC fields
    // span {16,32} kHz and {10,20,60,120} ms. The hard ceiling is therefore 32 * 120 = 3840 samples,
    // not the 960 of a common 60 ms / 16 kHz frame; a fuzzed TOC can legitimately declare a larger
    // frame, which the decoder fills with silence on the SID/inactive/std-opus paths.
    #[test]
    fn fuzz_decode_no_panic_bounded_output() {
        const MAX_SAMPS: usize = 32 * 120; // max sample_rate(kHz) * max frame_ms across all TOCs
        let mut dec = MlowDecoder::new();
        let check = |dec: &mut MlowDecoder, input: &[u8]| {
            let out = dec.decode(input);
            assert!(
                out.len() <= MAX_SAMPS,
                "decode emitted {} > {MAX_SAMPS} samples for input len {}",
                out.len(),
                input.len()
            );
        };

        // Deterministic LCG (numerical-recipes constants) over thousands of random-length buffers.
        let mut seed: u32 = 0x1234_5678;
        let next = |s: &mut u32| {
            *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *s
        };
        for _ in 0..8000 {
            let len = (next(&mut seed) % 400) as usize;
            let mut buf = Vec::with_capacity(len);
            for _ in 0..len {
                buf.push((next(&mut seed) >> 24) as u8);
            }
            check(&mut dec, &buf);
        }

        // Mutations of the real capture frames: every single-byte flip and every truncation.
        let frames: Vec<String> =
            serde_json::from_str(include_str!("testdata/inbound_capture_frames.json"))
                .expect("inbound_capture_frames.json");
        for hex_frame in &frames {
            let frame = hex::decode(hex_frame).unwrap();
            for i in 0..frame.len() {
                for bit in 0..8 {
                    let mut m = frame.clone();
                    m[i] ^= 1 << bit;
                    check(&mut dec, &m);
                }
                check(&mut dec, &frame[..i]); // truncation at every prefix length
            }
            check(&mut dec, &frame);
        }
    }

    // Fail-loud guards: a frame outside our single operating point (32kHz/fullband, or low_rate=1) is
    // DROPPED to 60ms silence WITHOUT touching the range coder, so it can't desync + poison the
    // cross-frame predictor. A real low_rate=0 frame before and after the drops still decodes (proves
    // the drop is a clean "lost frame", not a desync), and the guards don't false-positive on it.
    #[test]
    fn unsupported_frames_drop_clean_without_desync() {
        let frames: Vec<String> =
            serde_json::from_str(include_str!("testdata/inbound_capture_frames.json")).unwrap();
        let real = hex::decode(&frames[0]).unwrap();
        let mut dec = MlowDecoder::new();

        // A real 0x50 (16kHz, low_rate=0) frame decodes to audio — the guards don't false-positive.
        let normal = dec.decode(&real);
        assert_eq!(normal.len(), 960);
        assert!(
            normal.iter().any(|&s| s != 0.0),
            "a real low_rate=0 frame must decode to audio"
        );

        // A 32kHz/fullband TOC (bit5=1, e.g. 0x70) -> dropped to 60ms silence.
        let out_32k = dec.decode(&[0x70, 0xAA, 0xBB, 0xCC]);
        assert_eq!(out_32k.len(), 960);
        assert!(
            out_32k.iter().all(|&s| s == 0.0),
            "32kHz frame must drop to silence"
        );

        // A low_rate=1 TOC -> dropped to 60ms silence. 0x54 = 0b0101_0100: bit5=0 (16kHz, so the rate
        // branch does NOT catch it), bits4:3=10 (60ms), bit2=1 (low_rate) -> exercises the low_rate branch.
        let out_lr = dec.decode(&[0x54, 0xAA, 0xBB, 0xCC]);
        assert_eq!(out_lr.len(), 960);
        assert!(
            out_lr.iter().all(|&s| s == 0.0),
            "low_rate=1 frame must drop to silence"
        );

        // A non-60ms ACTIVE 16kHz/low_rate=0 TOC (20ms, e.g. 0x48) must also drop: decode_active_frame
        // hardcodes the 3x20ms / 60ms geometry, so a 20ms frame would otherwise desync the range coder.
        let out_20ms = dec.decode(&[0x48, 0xAA, 0xBB, 0xCC]);
        assert_eq!(out_20ms.len(), 960);
        assert!(
            out_20ms.iter().all(|&s| s == 0.0),
            "a non-60ms active frame must drop to silence"
        );

        // The drops never opened the range decoder, so the predictor is intact: the real frame still
        // decodes to audio afterwards (no desync from the dropped frames in between).
        assert!(
            !dec.had_error(),
            "dropped frames must not touch the range decoder"
        );
        let after = dec.decode(&real);
        assert!(
            after.iter().any(|&s| s != 0.0),
            "a real frame after the drops must still decode (no poisoned predictor)"
        );
    }

    // An inactive / DTX frame (TOC 0x00: vad=false so active=false, 16kHz, low_rate=0, 10ms) is the
    // benign startup/comfort silence a real peer emits, not a desync hazard: inactive frames are
    // silenced without opening the range coder regardless of geometry. It must take the quiet DTX/SID
    // path (a full 60ms silence slot, range coder untouched) and must NOT count as an out-of-operating
    // -point drop, which is reserved for active frames that would have lost decodable audio.
    #[test]
    fn inactive_off_point_frame_is_silenced_not_dropped() {
        let frames: Vec<String> =
            serde_json::from_str(include_str!("testdata/inbound_capture_frames.json")).unwrap();
        let real = hex::decode(&frames[0]).unwrap();
        let mut dec = MlowDecoder::new();

        assert!(
            dec.decode(&real).iter().any(|&s| s != 0.0),
            "a real low_rate=0 frame must decode to audio"
        );

        // TOC 0x00 -> inactive: silenced via the DTX/SID path, not the operating-point drop.
        let inactive = dec.decode(&[0x00, 0xAA, 0xBB, 0xCC]);
        assert_eq!(
            inactive.len(),
            960,
            "an inactive frame still fills a 60ms slot"
        );
        assert!(
            inactive.iter().all(|&s| s == 0.0),
            "an inactive frame must be silence"
        );
        assert_eq!(
            dec.dropped_unsupported, 0,
            "an inactive frame is the DTX/silence path, not an operating-point drop"
        );
        assert!(
            !dec.had_error(),
            "the inactive path must not open the range decoder"
        );

        // Contrast: an active off-point frame (0x48 = vad=true, 20ms) IS counted, proving the drop
        // counter discriminates real audio loss from benign inactive silence.
        let _ = dec.decode(&[0x48, 0xAA, 0xBB, 0xCC]);
        assert_eq!(
            dec.dropped_unsupported, 1,
            "an active off-point frame must count as a drop"
        );

        // After both the inactive frame and the drop, a real frame still decodes: no desync, no
        // poisoned predictor.
        assert!(
            dec.decode(&real).iter().any(|&s| s != 0.0),
            "a real frame must still decode after an inactive frame and a drop"
        );
    }

    // R7 (RED round-trip): a bare frame wrapped in a 1-redundant SplitRed envelope must decode to the
    // exact same PCM as the bare frame at redundancy 0. Exercises the `redundancy > 0` strip path
    // (which forwards the main/last frame) end-to-end.
    #[test]
    fn red_envelope_decodes_to_bare_main() {
        let frames: Vec<String> =
            serde_json::from_str(include_str!("testdata/inbound_capture_frames.json"))
                .expect("inbound_capture_frames.json");
        let bare = hex::decode(&frames[0]).unwrap();

        let mut bare_dec = MlowDecoder::new();
        let bare_out = bare_dec.decode(&bare);

        // SplitRed N=1: red_hdr [0x80 | tc, size], main_marker (high bit clear), red payload, main.
        // The main (last) frame is the bare frame, so the strip path must reproduce `bare_out`.
        let red_payload = [0xAAu8, 0xBB];
        let mut env = vec![0x80u8, red_payload.len() as u8, 0x00];
        env.extend_from_slice(&red_payload);
        env.extend_from_slice(&bare);

        let mut red_dec = MlowDecoder::new();
        red_dec.set_redundancy(1);
        let red_out = red_dec.decode(&env);

        assert_eq!(
            red_out, bare_out,
            "RED-wrapped main differs from bare decode"
        );
        assert!(
            !red_dec.had_error(),
            "RED decode raised the range decoder error flag"
        );
    }

    fn tone_frames(n: usize) -> Vec<Vec<u8>> {
        let mut encoder = super::super::encode::MlowEncoder::new();
        let tone: Vec<f32> = (0..OPUS_FRAME_SAMPS)
            .map(|i| 0.3 * (i as f32 * 0.05).sin())
            .collect();
        (0..n)
            .map(|_| encoder.encode(&tone).expect("mlow encode"))
            .collect()
    }

    fn envelope_of(frames: &[Vec<u8>]) -> Vec<u8> {
        assert!(frames.iter().all(|f| f.len() < 252));
        let mut envelope = vec![0x82 | (frames[0][0] & 0x39), frames.len() as u8];
        envelope.extend(frames[..frames.len() - 1].iter().map(|f| f.len() as u8));
        for f in frames {
            envelope.extend_from_slice(f);
        }
        envelope
    }

    // The bug this replaces: read as a TOC, the indicator has bit 7 set and is a SID, so every packet
    // of a call whose peer sends envelopes (a 0x92 first byte) became comfort silence.
    #[test]
    fn a_multiframe_envelope_decodes_instead_of_becoming_silence() {
        let frames = tone_frames(2);
        let envelope = envelope_of(&frames);
        assert_eq!(
            envelope[0], 0x92,
            "an inner 0x50 TOC gives the 0x92 indicator"
        );
        assert!(parse_mlow_toc(envelope[0]).sid);

        let mut single = MlowDecoder::new();
        let one = single.decode(&frames[0]);
        assert_eq!(one.len(), OPUS_FRAME_SAMPS);
        assert!(!single.last_was_multiframe());

        let mut decoder = MlowDecoder::new();
        let pcm = decoder.decode(&envelope);
        assert!(decoder.last_was_multiframe());
        assert_eq!(pcm.len(), 2 * OPUS_FRAME_SAMPS);
        assert!(
            pcm.iter().any(|&s| s != 0.0),
            "a decoded envelope must not be silence"
        );
        assert_eq!(
            &pcm[..OPUS_FRAME_SAMPS],
            one.as_slice(),
            "the first sub-frame decodes exactly as it would alone"
        );
        assert_eq!(decoder.dropped_unsupported, 0);
        assert!(!decoder.had_error());
    }

    #[test]
    fn a_true_sid_is_still_silence() {
        for toc in [0x90u8, 0x94, 0x98] {
            let mut decoder = MlowDecoder::new();
            let pcm = decoder.decode(&[toc, 0x01, 0x02]);
            assert_eq!(pcm.len(), OPUS_FRAME_SAMPS, "TOC {toc:#04x}");
            assert!(pcm.iter().all(|&s| s == 0.0), "TOC {toc:#04x}");
            assert!(!decoder.last_was_multiframe());
        }
    }

    #[test]
    fn a_malformed_multiframe_envelope_is_concealed_without_panicking() {
        let frames = tone_frames(2);
        let good = envelope_of(&frames);
        let mut bad: Vec<Vec<u8>> = vec![
            vec![0x92],
            vec![0x92, 0x02],
            vec![0x92, 0x02, 5],
            // two frames declared, the first taking the whole body
            [vec![0x92u8, 0x02, 10], (0..10u8).collect()].concat(),
            // a length that runs past the end
            [vec![0x92u8, 0x02, 200], (0..10u8).collect()].concat(),
            // frame count outside 1..=18
            [vec![0x92u8, 0x3f], (0..40u8).collect()].concat(),
        ];
        // every truncation of a real envelope
        bad.extend((1..good.len()).map(|n| good[..n].to_vec()));
        let mut decoder = MlowDecoder::new();
        for packet in &bad {
            let pcm = decoder.decode(packet);
            assert!(!pcm.is_empty());
            assert!(pcm.len() <= MAX_PACKET_SAMPS, "{} samples", pcm.len());
        }
        let pcm = decoder.decode(&[0x92, 0x02, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(pcm.len(), OPUS_FRAME_SAMPS);
        assert!(pcm.iter().all(|&s| s == 0.0));
        assert!(!decoder.last_was_multiframe());
    }

    #[test]
    fn an_envelope_past_the_duration_ceiling_is_concealed() {
        let frames = tone_frames(1);
        // three 60 ms sub-frames = 180 ms, over the 120 ms ceiling
        let envelope = envelope_of(&[frames[0].clone(), frames[0].clone(), frames[0].clone()]);
        let mut decoder = MlowDecoder::new();
        let pcm = decoder.decode(&envelope);
        assert_eq!(pcm.len(), OPUS_FRAME_SAMPS);
        assert!(pcm.iter().all(|&s| s == 0.0));
        assert!(!decoder.last_was_multiframe());
    }

    // The frame inside a RED wrapper can itself be an envelope.
    #[test]
    fn an_envelope_inside_a_red_wrapper_decodes() {
        let envelope = envelope_of(&tone_frames(2));
        let mut payload = vec![0x80u8, 2, 0x00, 0xAA, 0xBB];
        payload.extend_from_slice(&envelope);
        let mut decoder = MlowDecoder::new();
        decoder.set_redundancy(1);
        let pcm = decoder.decode(&payload);
        assert!(decoder.last_was_multiframe());
        assert_eq!(pcm.len(), 2 * OPUS_FRAME_SAMPS);
        assert!(pcm.iter().any(|&s| s != 0.0));
    }
}
