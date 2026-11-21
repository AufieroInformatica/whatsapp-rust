#!/usr/bin/env python3
"""Add an env-gated per-frame dump (SMPL_DUMP=<path>) to a scratch copy of smpl_core_decoder.c.
Only the scratch clone is touched; nothing here is committed anywhere."""
import sys

p = sys.argv[1]
s = open(p).read()

helpers = r'''
#include <stdio.h>
#include <stdlib.h>
static FILE *g_dump = NULL;
static int g_dump_init = 0;
static int g_dump_pkt = -1;
static FILE *dump_fp(void) {
    if (!g_dump_init) {
        g_dump_init = 1;
        const char *e = getenv("SMPL_DUMP");
        if (e) g_dump = fopen(e, "w");
    }
    return g_dump;
}
static void dump_f(FILE *f, const char *k, const float *v, int n) {
    fprintf(f, "{\"k\":\"%s\",\"v\":[", k);
    for (int i = 0; i < n; i++) fprintf(f, "%s%.9g", i ? "," : "", v[i]);
    fprintf(f, "]}\n");
}
static void dump_i(FILE *f, const char *k, const int *v, int n) {
    fprintf(f, "\"%s\":[", k);
    for (int i = 0; i < n; i++) fprintf(f, "%s%d", i ? "," : "", v[i]);
    fprintf(f, "]");
}
'''
s = s.replace('#include "silk/SigProc_FIX.h"\n', '#include "silk/SigProc_FIX.h"\n' + helpers, 1)

# 1) params dump after normalized bitrate computed
anchor = '        float normalized_bitrate = smpl_get_normalized_bitrate(lb_params.n_pulses, frame_length_16);\n'
assert anchor in s
dump_params = anchor + r'''        {
            FILE *df = dump_fp();
            if (df) {
                int tmp[64]; int n;
                if (frame == 0) g_dump_pkt++;
                fprintf(df, "{\"k\":\"params\",\"pkt\":%d,\"frame\":%d,\"voiced\":%d,\"nsf\":%d,\"interp\":%d,\"n_pulses\":%d,\"npos\":%d,",
                    g_dump_pkt, frame, lb_params.voiced, num_subframes, lb_params.lsf_interpol_idx, lb_params.n_pulses, lb_params.nPositions);
                for (n = 0; n < 17; n++) tmp[n] = lb_params.lsf_idx[n];
                dump_i(df, "lsf_idx", tmp, 17); fprintf(df, ",");
                for (n = 0; n < num_subframes; n++) tmp[n] = lb_params.sf_pulses[n];
                dump_i(df, "sf_pulses", tmp, num_subframes); fprintf(df, ",");
                for (n = 0; n < lb_params.nPositions; n++) tmp[n] = lb_params.positions[n];
                dump_i(df, "positions", tmp, lb_params.nPositions); fprintf(df, ",");
                for (n = 0; n < lb_params.nPositions; n++) tmp[n] = lb_params.pos_pulses[n];
                dump_i(df, "pos_pulses", tmp, lb_params.nPositions); fprintf(df, ",");
                for (n = 0; n < num_subframes; n++) tmp[n] = lb_params.acbg_idx[n];
                dump_i(df, "acbg_idx", tmp, num_subframes); fprintf(df, ",");
                for (n = 0; n < num_subframes; n++) tmp[n] = lb_params.fcbg_idx[n];
                dump_i(df, "fcbg_idx", tmp, num_subframes); fprintf(df, ",");
                for (n = 0; n < num_subframes; n++) tmp[n] = lb_params.nrgres_dbq_Q14[n];
                dump_i(df, "nrgres_dbq_q14", tmp, num_subframes); fprintf(df, ",");
                for (n = 0; n < lags_per_frame; n++) tmp[n] = lb_params.laginds[n];
                dump_i(df, "laginds", tmp, lags_per_frame);
                fprintf(df, ",\"tell\":%d}\n", (int)ec_tell(psRangeDec));
                dump_f(df, "lsfq", lsfq, SMPL_LPC_ORDER);
                for (n = 0; n < num_subframes; n++) dump_f(df, "lsfs", lsfs[n], SMPL_LPC_ORDER);
                for (n = 0; n < num_subframes; n++) dump_f(df, "A", A[n], SMPL_LPC_ORDER + 1);
            }
        }
'''
s = s.replace(anchor, dump_params, 1)

# 2) excitation after celp decode (pre-noise) and noise
a2 = '            TOC(celp)\n'
assert a2 in s
s = s.replace(a2, a2 + r'''            { FILE *df = dump_fp(); if (df) dump_f(df, "exc_pre", &lpc_res[sf * subframe_length_16], subframe_length_16); }
''', 1)
a3 = '            TOC(gen_noise)\n\n#ifdef SMPL_USE_TILT_POSTFILTER'
assert a3 in s
s = s.replace(a3, r'''            { FILE *df = dump_fp(); if (df) dump_f(df, "noise", noise, subframe_length_16); }
''' + a3, 1)

# 3) y after the subframe loop (pre postfilters)
a4 = '        memcpy(dec_state->lpc_synth_mem, &y[frame_length_16 - SMPL_LPC_ORDER], SMPL_LPC_ORDER * sizeof(float));\n        TOC(subfr_loop)\n'
assert a4 in s
s = s.replace(a4, a4 + r'''        { FILE *df = dump_fp(); if (df) dump_f(df, "y_synth", y, frame_length_16); }
''', 1)

# 4) y after HP postfilter
a5 = '        smpl_assert(!res);\n        TOC(hp_postfilter)\n'
assert a5 in s
s = s.replace(a5, a5 + r'''        { FILE *df = dump_fp(); if (df) dump_f(df, "y_hp", y, frame_length_16); }
''', 1)

# 5) packet after harm postfilter
a6 = '    TOC(harm_postfilter)\n#endif\n'
assert a6 in s
s = s.replace(a6, a6 + r'''    { FILE *df = dump_fp(); if (df) { dump_f(df, "y_out", yBuf, packetlen_16); fflush(df); } }
''', 1)
open(p, 'w').write(s)
print("patched")
