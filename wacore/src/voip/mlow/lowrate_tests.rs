//! End-to-end checks for the reduced-rate (TOC bit 2) decode path.
//!
//! The vectors are packets from the pinned C reference encoder (`edgardmessias/opus_mlow` at
//! `84b076e0`, the same revision the existing C audits use) run at 8-10 kbps on the synthetic
//! `synth_mic.raw`, paired with what the C reference decoder produced for them. No recorded call
//! audio is involved. See `testdata/PROVENANCE.md` for how they were produced and what they do and
//! do not establish.

#![cfg(test)]

use super::decoder::MlowDecoder;

fn frames(json: &str) -> Vec<Vec<u8>> {
    let hex: Vec<String> = serde_json::from_str(json).expect("fixture frames");
    hex.iter()
        .map(|h| hex::decode(h).expect("fixture hex"))
        .collect()
}

fn pcm(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
        .collect()
}

fn corr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
        b.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
    );
    let (mut sxy, mut sxx, mut syy) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (f64::from(x) - ma, f64::from(y) - mb);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx * syy).sqrt()
}

/// Decode every packet through one decoder, as a call would, and return the PCM plus how many
/// packets the decoder refused or concealed.
fn decode_stream(packets: &[Vec<u8>]) -> (Vec<f32>, u32, u32) {
    let mut dec = MlowDecoder::new();
    let mut out = Vec::new();
    let (mut off_point, mut concealed) = (0u32, 0u32);
    for p in packets {
        out.extend_from_slice(&dec.decode(p));
        let r = dec.take_frame_report();
        off_point += u32::from(r.off_point);
        concealed += u32::from(r.concealed);
    }
    (out, off_point, concealed)
}

/// Decode a pure low-rate stream and hold it to a reference decode of the same packets.
///
/// The thresholds sit under what is measured (corr 0.9998, 31-37 dB) with a margin for libm
/// differences, and well above where a wrong codebook or filter lands (a missing tilt filter or a
/// wrong subframe split drops the SNR below 15 dB).
fn check_against_reference(frames_json: &str, ref_raw: &[u8], min_corr: f64, min_snr_db: f64) {
    let packets = frames(frames_json);
    assert!(
        packets.iter().all(|p| p[0] & 0x04 != 0),
        "the fixture must stay a pure low-rate stream"
    );
    let refp = pcm(ref_raw);
    let (out, off_point, concealed) = decode_stream(&packets);
    assert_eq!(off_point, 0, "no low-rate packet may be refused");
    assert_eq!(concealed, 0, "every low-rate packet must parse to its end");
    assert_eq!(out.len(), refp.len(), "decoded length vs reference");
    assert!(
        out.iter().any(|&s| s != 0.0),
        "a low-rate stream must not decode to silence"
    );
    let c = corr(&refp, &out);
    let (mut sig, mut err) = (0f64, 0f64);
    for (&r, &o) in refp.iter().zip(&out) {
        sig += f64::from(r) * f64::from(r);
        err += (f64::from(r) - f64::from(o)).powi(2);
    }
    let snr_db = 10.0 * (sig / err.max(1e-30)).log10();
    assert!(c > min_corr, "lag-0 corr {c:.6} vs the reference");
    assert!(snr_db > min_snr_db, "SNR {snr_db:.1} dB vs the reference");
}

#[test]
fn low_rate_60ms_decode_matches_the_c_reference() {
    check_against_reference(
        include_str!("testdata/lr_c_lr60_frames.json"),
        include_bytes!("testdata/lr_c_lr60_ref.raw"),
        0.999,
        30.0,
    );
}

#[test]
fn low_rate_20ms_decode_matches_the_c_reference() {
    check_against_reference(
        include_str!("testdata/lr_c_lr20_frames.json"),
        include_bytes!("testdata/lr_c_lr20_ref.raw"),
        0.999,
        30.0,
    );
}

#[test]
fn low_rate_120ms_decode_matches_the_c_reference() {
    check_against_reference(
        include_str!("testdata/lr_c_lr120_frames.json"),
        include_bytes!("testdata/lr_c_lr120_ref.raw"),
        0.999,
        30.0,
    );
}

/// The shipped VoIP module's own encoder at an 8 kbps target (`SET_BITRATE`), and what its own
/// decoder made of those packets. Independent of the C reference above: it is the vendor binary on
/// both sides.
#[test]
fn low_rate_60ms_decode_matches_the_shipped_wasm() {
    check_against_reference(
        include_str!("testdata/lr_wasm_lr60_frames.json"),
        include_bytes!("testdata/lr_wasm_lr60_ref.raw"),
        0.999,
        28.0,
    );
}

/// The same module at a lower target (6 kbps), so the pulse budgets are smaller and the parse
/// reaches different branches.
#[test]
fn low_rate_decode_matches_the_shipped_wasm_at_6kbps() {
    check_against_reference(
        include_str!("testdata/lr_wasm_lr60_6k_frames.json"),
        include_bytes!("testdata/lr_wasm_lr60_6k_ref.raw"),
        0.999,
        28.0,
    );
}

/// 120 ms packets (six chained internal frames) from the shipped module at the low rate.
#[test]
fn low_rate_120ms_decode_matches_the_shipped_wasm() {
    check_against_reference(
        include_str!("testdata/lr_wasm_lr120_frames.json"),
        include_bytes!("testdata/lr_wasm_lr120_ref.raw"),
        0.999,
        28.0,
    );
}

/// A low-rate packet whose body cannot hold its frames is concealed as lost, reported as concealed
/// rather than refused, and leaves nothing behind: the next real packet decodes exactly as it does
/// on a decoder that never saw the bad one.
#[test]
fn a_malformed_low_rate_packet_is_concealed_and_rolled_back() {
    let packets = frames(include_str!("testdata/lr_c_lr60_frames.json"));
    let mut with_bad = MlowDecoder::new();
    let mut clean = MlowDecoder::new();
    for p in &packets[..6] {
        let _ = with_bad.decode(p);
        let _ = clean.decode(p);
    }
    let _ = with_bad.take_frame_report();

    let bad = with_bad.decode(&[0x54, 0xAA, 0xBB, 0xCC]);
    let report = with_bad.take_frame_report();
    assert_eq!(bad.len(), 960);
    assert!(bad.iter().all(|&s| s == 0.0), "a malformed body is silence");
    assert_eq!(
        (report.concealed, report.off_point, report.decoded),
        (1, 0, 0)
    );

    for p in &packets[6..12] {
        assert_eq!(
            with_bad.decode(p),
            clean.decode(p),
            "a concealed packet must not leak state into the next one"
        );
    }
}

/// Comfort-noise frames at the low rate (TOC 0x94 is the one seen on calls) never open the range
/// coder, same as at the full rate.
#[test]
fn low_rate_sid_frames_are_silence_not_a_refusal() {
    let mut dec = MlowDecoder::new();
    for toc in [0x94u8, 0x9C] {
        let out = dec.decode(&[toc, 0x12, 0x34]);
        let report = dec.take_frame_report();
        assert!(out.iter().all(|&s| s == 0.0), "TOC {toc:#04x}");
        assert_eq!(
            (report.inactive_or_sid, report.off_point, report.concealed),
            (1, 0, 0),
            "TOC {toc:#04x}"
        );
    }
}

/// What a phone does mid-call: full-rate voice, then the switch to the low rate, never back. By
/// the end of the low-rate stream the decoder that came from full rate must be producing the same
/// audio as one that started cold, once the history carried over from the other geometry has
/// turned over.
#[test]
fn a_call_that_switches_to_low_rate_keeps_decoding() {
    let hr = frames(include_str!("testdata/wasm_derived_frames.json"));
    let lr = frames(include_str!("testdata/lr_c_lr60_frames.json"));

    let mut dec = MlowDecoder::new();
    let (mut audible_hr, mut audible_lr) = (0usize, 0usize);
    for p in hr.iter().take(40) {
        let out = dec.decode(p);
        audible_hr += usize::from(out.iter().any(|&s| s != 0.0));
    }
    let mut switched = Vec::new();
    for p in &lr {
        let out = dec.decode(p);
        let r = dec.take_frame_report();
        assert_eq!(r.off_point + r.concealed, 0, "no packet lost at the switch");
        assert!(out.iter().all(|s| s.abs() <= 1.0 && s.is_finite()));
        audible_lr += usize::from(out.iter().any(|&s| s != 0.0));
        switched.extend_from_slice(&out);
    }
    assert!(audible_hr > 30, "the full-rate half must be audible");
    assert!(audible_lr > 100, "the low-rate half must be audible");

    let (cold, _, _) = decode_stream(&lr);
    let tail = 30 * 960;
    let c = corr(
        &cold[cold.len() - tail..],
        &switched[switched.len() - tail..],
    );
    assert!(c > 0.99, "tail correlation after a rate switch {c:.5}");
}

/// Adversarial input at the low rate: truncations, byte flips and random bodies under low-rate TOCs
/// must neither panic nor produce more audio than a TOC can declare.
#[test]
fn low_rate_decode_survives_corrupt_input() {
    let packets = frames(include_str!("testdata/lr_c_lr60_frames.json"));
    let mut dec = MlowDecoder::new();
    let mut decoded = 0u32;
    let mut check = |dec: &mut MlowDecoder, bytes: &[u8]| {
        let out = dec.decode(bytes);
        assert!(out.len() <= 32 * 120, "bounded output");
        assert!(out.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
        decoded += 1;
    };
    for p in &packets {
        for cut in 0..p.len() {
            check(&mut dec, &p[..cut]);
        }
        for i in 0..p.len() {
            let mut flipped = p.clone();
            flipped[i] ^= 0xFF;
            check(&mut dec, &flipped);
        }
    }
    let mut seed = 0x1234_5678u32;
    for n in 0..4000 {
        let mut body = vec![0x54 | (n as u8 & 0xA0)];
        for _ in 0..(n % 90) + 1 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            body.push((seed >> 24) as u8);
        }
        check(&mut dec, &body);
    }
    let expected: usize = packets.iter().map(|p| 2 * p.len()).sum::<usize>() + 4000;
    assert_eq!(
        decoded as usize, expected,
        "every corrupt input must have been decoded"
    );
}
