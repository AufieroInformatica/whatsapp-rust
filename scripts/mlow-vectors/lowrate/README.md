# Reduced-rate (low_rate) vectors

How the `lr_c_*` and `lr_wasm_*` fixtures in `wacore/src/voip/mlow/testdata` were made, and how to
redo them. Both are driven by the synthetic `synth_mic.raw`; there is no recorded audio here.

## From the C reference

1. Check out `edgardmessias/opus_mlow` at `84b076e0809412df22e8a0d26f944610c4a3e40f` (the revision
   the other C audits pin) and build it: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   -DOPUS_BUILD_PROGRAMS=OFF -DOPUS_BUILD_TESTING=OFF && cmake --build build`.
2. `python3 c_param_dump.py <checkout>/smpl/smpl_core_decoder.c` adds an environment-gated dump
   (`SMPL_DUMP=<file>`) of the decoded parameters and the synthesis output per internal frame, then
   rebuild. The dump does nothing unless the variable is set.
3. Compile `../mlow_frames.rs` against `build/libopus.a` (see the note in that file; the rate is
   chosen by the bitrate argument).
4. `REF_BIN=<that binary> SCRATCH=<dir> python3 gen_c_lowrate.py`.

The reference picks the reduced rate when the target bitrate is at or under 8700 bps for 60 and
120 ms packets, and 10200 bps for 20 ms packets. The 60 ms stream at 8 kbps produces TOC bytes
`0x54`, `0x16` and `0x14`, the same family a phone sends once a call goes to video.

## From the shipped module

`wasm_lowrate.py` rewrites an existing derivation spec so the module's own encoder runs at a low
bitrate (`OPUS_SET_BITRATE` through its `encoder_ctl`), then packs what `oracle derive` wrote. At
8000 bps and 60 ms the module emits `0x54`, `0x16` and `0x14` packets, and its decoder accepts them.
See the script's docstring for the commands. The capture must be present (`cargo xt mlow fetch`).

## What the vectors do and do not establish

They show that this decoder reads reduced-rate packets from the C reference encoder and from the
shipped module's encoder the way the C reference decoder and the shipped module's decoder do. The
packets are from synthetic input at 6 to 10 kbps. They do not include packets captured from a phone.
