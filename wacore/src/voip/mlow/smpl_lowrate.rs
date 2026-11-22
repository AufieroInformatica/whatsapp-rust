//! Reduced-rate frame parameters (TOC bit 2 set): two 160-sample subframes per 20 ms internal
//! frame instead of four of 80.
//!
//! Everything up to the pulse positions is shared with the full-rate path and already takes the
//! subframe count: the LSF block reads its codebooks by the rate bit, the pulse block has a
//! two-subframe split (`smpl_pulse`), and the pitch block loops over `p3` subframes. What differs
//! is the unvoiced energy-residual block, which has its own two-subframe codebooks. Those come from
//! the pinned C reference (`smpl/smpl_nrgres_tables.c` at `84b076e0`: `smpl_nrg_step_db_Q14[1]`,
//! `nrgres_shape_CB_2_Q10`, `nrgres_shape_CB_2_dcmf`, `smpl_nrgres_gain_2_dcmf`) rather than from the
//! shipped module, because the vendor oracle had not been run on a reduced-rate stream when they
//! were transcribed.

use std::sync::OnceLock;

use super::rangecoder::RangeDecoder;
use super::smpl_cc_tables::CcTables;
use super::smpl_celp::dcmf_to_cmf;
use super::smpl_celpdec::CelpDecParams;
use super::smpl_decode::{SmplLsfIndices, SmplLsfState, SmplTables, decode_smpl_lsf};
use super::smpl_mem::SmplMem;
use super::smpl_pitch::decode_smpl_pitch;
use super::smpl_pulse::{SmplPulseResult, decode_smpl_pulses};

/// Internal-frame length in samples (20 ms at 16 kHz), unchanged by the rate bit.
const FRAME_LEN: i32 = 320;
/// Subframes per internal frame at the reduced rate.
const NUM_SUBFR: i32 = 2;
/// Index of the two-subframe row in the three-row (1, 2, 4 subframes) fcbg-offset tables.
const TABLE_IX_2: usize = 1;
const RES_NRG_MIN_DB: i32 = -85;
const RES_NRG_MAX_DB: i32 = 0;

const NRG_STEP_DB_Q14: [i32; 3] = [16384, 19843, 16686];
const NRGRES_SHAPE_CB_2_Q10: [i32; 44] = [
    -9672, 9672, -4437, 4437, -907, 907, 3610, -3610, -2857, 2857, 9345, -9345, -2220, 2220, 420,
    -420, -3565, 3565, -1577, 1577, -184, 184, 2973, -2973, 2389, -2389, 1774, -1774, -7315, 7315,
    4405, -4405, -5630, 5630, -12927, 12927, -17376, 17376, 1107, -1107, 7058, -7058, 5481, -5481,
];
const NRGRES_SHAPE_CB_2_DCMF: [u8; 22] = [
    47, 52, 137, 58, 61, 21, 73, 240, 55, 95, 255, 68, 83, 108, 48, 50, 49, 45, 35, 155, 31, 41,
];
const NRGRES_GAIN_2_DCMF: [u8; 71] = [
    139, 194, 196, 200, 209, 225, 239, 243, 246, 246, 253, 255, 253, 249, 244, 239, 233, 231, 230,
    228, 219, 216, 214, 214, 213, 211, 214, 215, 220, 223, 230, 234, 237, 240, 242, 244, 245, 245,
    246, 242, 237, 230, 220, 210, 190, 170, 148, 125, 104, 83, 64, 49, 36, 22, 14, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
];

struct NrgTables {
    gain: Vec<u16>,
    shape: Vec<u16>,
}

fn nrg_tables() -> &'static NrgTables {
    static T: OnceLock<NrgTables> = OnceLock::new();
    T.get_or_init(|| NrgTables {
        gain: dcmf_to_cmf(&NRGRES_GAIN_2_DCMF),
        shape: dcmf_to_cmf(&NRGRES_SHAPE_CB_2_DCMF),
    })
}

/// Per-subframe unvoiced gains: the quantized residual-energy value and the FCB-gain offset symbol
/// (only subframes that carry pulses read one).
struct UnvoicedGains {
    nrgres_dbq_q14: [i32; 4],
    fcbg_idx: [i32; 4],
}

/// The unvoiced block of a two-subframe frame: a frame energy index, a shape index that splits it
/// across the subframes, then one FCB-gain offset per subframe that has pulses, coded relative to
/// that subframe's own energy.
fn decode_unvoiced_gains(
    dec: &mut RangeDecoder,
    cc: &CcTables,
    subfr_pulses: [i32; 4],
) -> UnvoicedGains {
    let t = nrg_tables();
    let frame_qi = dec.decode_cdf(&t.gain);
    let shape_qi = dec.decode_cdf(&t.shape) as usize;
    let frame_dbq_q14 = frame_qi * NRG_STEP_DB_Q14[TABLE_IX_2] + RES_NRG_MIN_DB * (1 << 14);
    let mut out = UnvoicedGains {
        nrgres_dbq_q14: [0; 4],
        fcbg_idx: [0; 4],
    };
    for sf in 0..NUM_SUBFR as usize {
        out.nrgres_dbq_q14[sf] = frame_dbq_q14 + NRGRES_SHAPE_CB_2_Q10[shape_qi * 2 + sf] * 16;
    }
    for sf in 0..NUM_SUBFR as usize {
        let n = subfr_pulses[sf];
        if n <= 0 {
            continue;
        }
        let db = ((out.nrgres_dbq_q14[sf] + (1 << 13)) >> 14).clamp(RES_NRG_MIN_DB, RES_NRG_MAX_DB);
        let bucket = ((n / 10) as usize).min(3);
        out.fcbg_idx[sf] = dec.decode_cdf(cc.fcbg_offset(TABLE_IX_2, bucket, (-db) as usize));
    }
    out
}

/// One decoded reduced-rate internal frame.
pub(crate) struct LowRateFrame {
    pub(crate) lsf: SmplLsfIndices,
    pub(crate) pulses: SmplPulseResult,
    pub(crate) params: CelpDecParams,
}

/// Decode the parameters of one reduced-rate internal frame. `intf` is the frame's index within
/// the packet, which selects the conditional-coding state in the LSF block.
pub(crate) fn decode_low_rate_frame(
    dec: &mut RangeDecoder,
    tbl: &SmplTables,
    mem: &SmplMem,
    cc: &CcTables,
    lstate: &mut SmplLsfState,
    intf: usize,
    coded_as_active_voice: bool,
) -> LowRateFrame {
    let lsf = decode_smpl_lsf(dec, tbl, lstate, 1, intf, coded_as_active_voice);
    let pulses = decode_smpl_pulses(
        dec,
        cc,
        FRAME_LEN,
        NUM_SUBFR,
        i32::from(coded_as_active_voice),
        1,
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
        let pr = decode_smpl_pitch(dec, mem, cc, lstate, FRAME_LEN, NUM_SUBFR, 1, pulses.subfr);
        for b in 0..8 {
            params.block_lags[b] = ((pr.block_lags[b] as f64 * 0.5 + 32.0).min(320.0)) as f32;
        }
        for sf in 0..NUM_SUBFR as usize {
            params.acbg_idx[sf] = pr.gain_idx[sf];
            params.fcbg_idx[sf] = pr.filt_idx[sf].max(0);
        }
    } else {
        let g = decode_unvoiced_gains(dec, cc, pulses.subfr);
        params.nrgres_dbq_q14 = g.nrgres_dbq_q14;
        params.fcbg_idx = g.fcbg_idx;
    }
    LowRateFrame {
        lsf,
        pulses,
        params,
    }
}

#[cfg(test)]
mod tests {
    use super::super::smpl_cc_tables::load_cc_tables;
    use super::super::smpl_decode::load_smpl_tables;
    use super::super::smpl_mem::load_smpl_mem;
    use super::super::smpl_synth::{load_smpl_synth_tables, smpl_reconstruct_nlsf};
    use super::super::toc::parse_mlow_toc;
    use super::*;
    use serde_json::Value;

    fn ints(v: &Value) -> Vec<i32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect()
    }

    /// The synthesis stage on its own, frame by frame against the C reference's output for the
    /// same frame (after the LPC synthesis, tilt filter and HP postfilter, before the per-packet
    /// harmonic postfilter). The end-to-end SNR is capped near 35 dB by the reference's own int16
    /// rounding, which hides small filter errors; per frame, at float precision, they show.
    #[test]
    fn synthesis_matches_the_c_reference_frame_by_frame() {
        use super::super::smpl_celpdec::CelpDecState;
        let packets: Vec<String> =
            serde_json::from_str(include_str!("testdata/lr_c_lr60_frames.json")).unwrap();
        let want: Vec<f32> = include_bytes!("testdata/lr_c_lr60_yhp.f32")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(want.len(), packets.len() * 3 * 320);

        let (tbl, mem, cc, synth) = (
            load_smpl_tables(),
            load_smpl_mem(),
            load_cc_tables(),
            load_smpl_synth_tables(),
        );
        let mut lstate = SmplLsfState::default();
        let mut celp = CelpDecState::default();
        let mut prev_nlsf: Vec<f32> = Vec::new();
        let mut worst = (0f32, 0usize, 0usize);
        for (pkt, hex_frame) in packets.iter().enumerate() {
            let frame = hex::decode(hex_frame).unwrap();
            let toc = parse_mlow_toc(frame[0]);
            let mut dec = RangeDecoder::new(&frame[1..]);
            for f in 0..3 {
                let fr = decode_low_rate_frame(&mut dec, tbl, mem, cc, &mut lstate, f, toc.active);
                let nlsf = smpl_reconstruct_nlsf(
                    synth,
                    fr.lsf.stage1 as usize,
                    1,
                    fr.lsf.grid as usize,
                    &fr.lsf.stage2,
                    &prev_nlsf,
                );
                let mut sig = [0f32; 320];
                celp.synth_frame_lr(
                    &nlsf,
                    fr.lsf.extra as usize,
                    &fr.pulses.pulses,
                    &fr.params,
                    &mut sig,
                );
                prev_nlsf = nlsf;
                let base = (pkt * 3 + f) * 320;
                for (i, (&a, &b)) in sig.iter().zip(&want[base..base + 320]).enumerate() {
                    let d = (a - b).abs();
                    if d > worst.0 {
                        worst = (d, pkt * 3 + f, i);
                    }
                }
            }
        }
        // Peaks are about 0.2. The port's float math is not the reference's (the same few-percent
        // per-frame deviation shows in the full-rate comparison), so the bound is a few percent of
        // the peak. Measured worst case is 7.9e-3. Dropping the tilt filter, the unvoiced pulse
        // shaping or the pitch sharpening, or moving one LSF interpolation weight by 0.1, each
        // lands at 1.7e-2 or more.
        assert!(worst.0 < 1.2e-2, "worst synthesis error {worst:?}");
    }

    /// Every internal frame of the 60 ms reduced-rate stream, decoded exactly as the packet loop
    /// does, against the symbols the C reference decoder read out of the same bytes: voicing, LSF
    /// indices, pulse counts and signed positions, gains, lag indices, and where the range decoder
    /// stood afterwards. A wrong codebook or a wrong symbol count shows up here as the first field
    /// that disagrees, rather than as noise in the PCM.
    #[test]
    fn frame_parameters_match_the_c_reference() {
        let packets: Vec<String> =
            serde_json::from_str(include_str!("testdata/lr_c_lr60_frames.json")).unwrap();
        let recs: Value =
            serde_json::from_str(include_str!("testdata/lr_c_lr60_params.json")).unwrap();
        let recs = recs.as_array().unwrap();
        assert_eq!(recs.len(), packets.len() * 3);

        let (tbl, mem, cc, synth) = (
            load_smpl_tables(),
            load_smpl_mem(),
            load_cc_tables(),
            load_smpl_synth_tables(),
        );
        let mut lstate = SmplLsfState::default();
        let mut prev_nlsf: Vec<f32> = Vec::new();
        let (mut voiced_frames, mut unvoiced_frames, mut inactive_frames) = (0, 0, 0);
        for (pkt, hex_frame) in packets.iter().enumerate() {
            let frame = hex::decode(hex_frame).unwrap();
            let toc = parse_mlow_toc(frame[0]);
            assert!(toc.low_rate && !toc.sid && toc.frame_ms == 60);
            let mut dec = RangeDecoder::new(&frame[1..]);
            for f in 0..3 {
                let want = &recs[pkt * 3 + f];
                let at = format!("pkt {pkt} frame {f}");
                assert_eq!(want["pkt"].as_u64().unwrap() as usize, pkt, "{at}");
                let fr = decode_low_rate_frame(&mut dec, tbl, mem, cc, &mut lstate, f, toc.active);

                assert_eq!(
                    i64::from(fr.params.voiced),
                    want["voiced"].as_i64().unwrap(),
                    "voiced {at}"
                );
                let lsf_idx = ints(&want["lsf_idx"]);
                assert_eq!(fr.lsf.grid, lsf_idx[0], "lsf stage-1 index {at}");
                assert_eq!(fr.lsf.stage2.to_vec(), lsf_idx[1..], "lsf stage-2 {at}");
                if toc.active {
                    assert_eq!(
                        i64::from(fr.lsf.extra),
                        want["interp"].as_i64().unwrap(),
                        "lsf interpolation index {at}"
                    );
                }
                assert_eq!(
                    fr.pulses.subfr[..2].to_vec(),
                    ints(&want["sf_pulses"]),
                    "pulses per subframe {at}"
                );
                let (mut positions, mut counts) = (Vec::new(), Vec::new());
                for (pos, &v) in fr.pulses.pulses.iter().enumerate() {
                    if v != 0 {
                        positions.push(pos as i32);
                        counts.push(v);
                    }
                }
                assert_eq!(positions, ints(&want["positions"]), "pulse positions {at}");
                assert_eq!(
                    counts,
                    ints(&want["pos_pulses"]),
                    "signed pulse counts {at}"
                );
                if fr.params.voiced {
                    assert_eq!(
                        fr.params.acbg_idx[..2].to_vec(),
                        ints(&want["acbg_idx"]),
                        "acb gain index {at}"
                    );
                    let c_fcbg = ints(&want["fcbg_idx"]);
                    for sf in 0..2 {
                        if fr.params.sf_pulses[sf] > 0 {
                            assert_eq!(fr.params.fcbg_idx[sf], c_fcbg[sf], "fcb gain {at} sf{sf}");
                        }
                    }
                    let lags: Vec<i32> = fr
                        .params
                        .block_lags
                        .iter()
                        .map(|&l| ((l - 32.0) * 2.0).round() as i32)
                        .collect();
                    assert_eq!(lags, ints(&want["laginds"]), "lag indices {at}");
                    voiced_frames += 1;
                } else {
                    assert_eq!(
                        fr.params.nrgres_dbq_q14[..2].to_vec(),
                        ints(&want["nrgres_dbq_q14"]),
                        "residual energy {at}"
                    );
                    let c_fcbg = ints(&want["fcbg_idx"]);
                    for sf in 0..2 {
                        if fr.params.sf_pulses[sf] > 0 {
                            assert_eq!(
                                fr.params.fcbg_idx[sf], c_fcbg[sf],
                                "fcb offset {at} sf{sf}"
                            );
                        }
                    }
                    if toc.active {
                        unvoiced_frames += 1;
                    } else {
                        inactive_frames += 1;
                    }
                }
                assert_eq!(
                    dec.tell() as i64,
                    want["tell"].as_i64().unwrap(),
                    "range decoder position {at}"
                );
                assert_eq!(dec.err, 0, "range decoder error {at}");

                let nlsf = smpl_reconstruct_nlsf(
                    synth,
                    fr.lsf.stage1 as usize,
                    1,
                    fr.lsf.grid as usize,
                    &fr.lsf.stage2,
                    &prev_nlsf,
                );
                let c_nlsf: Vec<f32> = want["lsfq"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect();
                // The only place the two decoders part is the minimum-spacing clamp on an unvoiced
                // frame: this decoder's tables follow the shipped module's selection, which is the
                // reverse of the reference's voiced/unvoiced naming, so a clamped low LSF lands
                // where the other row puts it. The rows differ by up to 7e-3 per spacing and the
                // clamp carries that along the first few LSFs, so the bound for those frames is
                // loose (2e-2 against LSFs of order 0.01 to 3). Output PCM does not see it: the
                // shipped-module comparison is identical to four decimals with either selection.
                // Voiced frames agree to float precision.
                let tolerance = if fr.params.voiced { 1e-5 } else { 2e-2 };
                for (k, (a, b)) in nlsf.iter().zip(&c_nlsf).enumerate() {
                    assert!(
                        (a - b).abs() < tolerance,
                        "reconstructed NLSF[{k}] {a} vs {b} {at} stage1={} grid={} active={}\n got  {nlsf:?}\n want {c_nlsf:?}\n stage2 {:?}",
                        fr.lsf.stage1,
                        fr.lsf.grid,
                        toc.active,
                        fr.lsf.stage2
                    );
                }
                prev_nlsf = nlsf;
            }
        }
        assert!(
            voiced_frames > 100 && unvoiced_frames > 50 && inactive_frames > 0,
            "the fixture must keep exercising voiced ({voiced_frames}), unvoiced \
             ({unvoiced_frames}) and coded-inactive ({inactive_frames}) frames"
        );
    }
}
