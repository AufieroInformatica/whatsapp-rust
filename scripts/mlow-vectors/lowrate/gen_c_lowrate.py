#!/usr/bin/env python3
"""Regenerate the `lr_c_*` reduced-rate fixtures from the pinned C reference.

Needs the reference built (see README.md) with `c_param_dump.py` applied to its
`smpl/smpl_core_decoder.c`, and `mlow_frames` compiled against its libopus.a.

    REF_BIN=/path/to/mlow_frames SCRATCH=/some/dir python3 gen_c_lowrate.py

Input is the synthetic `synth_mic.raw`; no recorded audio is involved.
"""
import json, os, struct, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../../../wacore/src/voip/mlow/testdata")
BIN = os.environ["REF_BIN"]
SCRATCH = os.environ["SCRATCH"]
INPUT = os.path.join(DATA, "synth_mic.raw")

# (tag, packet ms, packets, bitrate). The reference chooses the reduced rate when the bitrate is at
# or under 8700 bps (60/120 ms) or 10200 bps (20 ms); these sit just below.
RUNS = [("lr60", 60, 110, 8000), ("lr20", 20, 150, 10000), ("lr120", 120, 8, 8500)]


def run(ms, count, bitrate, tag):
    dump = os.path.join(SCRATCH, f"{tag}.dump")
    env = dict(os.environ, SMPL_DUMP=dump)
    out = subprocess.run([BIN, INPUT, str(ms), str(count), str(bitrate)], env=env,
                         capture_output=True, text=True, check=True).stdout
    frames, pcm = [], bytearray()
    for line in out.splitlines():
        packet, decoded = line.split()
        frames.append(packet)
        pcm += bytes.fromhex(decoded)
    recs, cur, yhp = [], None, bytearray()
    for line in open(dump):
        d = json.loads(line)
        if d["k"] == "params":
            cur = d
            recs.append(cur)
        elif d["k"] == "lsfq":
            cur["lsfq"] = d["v"]
        elif d["k"] == "y_hp":
            yhp += struct.pack(f"<{len(d['v'])}f", *d["v"])
    return frames, bytes(pcm), recs, bytes(yhp)


def write_json(name, obj):
    with open(os.path.join(DATA, name), "w") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.write("\n")


for tag, ms, count, bitrate in RUNS:
    frames, pcm, recs, yhp = run(ms, count, bitrate, tag)
    write_json(f"lr_c_{tag}_frames.json", frames)
    with open(os.path.join(DATA, f"lr_c_{tag}_ref.raw"), "wb") as f:
        f.write(pcm)
    if tag == "lr60":
        keys = ("pkt", "frame", "voiced", "interp", "n_pulses", "lsf_idx", "sf_pulses", "positions",
                "pos_pulses", "acbg_idx", "fcbg_idx", "nrgres_dbq_q14", "laginds", "tell", "lsfq")
        write_json("lr_c_lr60_params.json", [{k: r[k] for k in keys} for r in recs])
        with open(os.path.join(DATA, "lr_c_lr60_yhp.f32"), "wb") as f:
            f.write(yhp)
    print(tag, len(frames), "packets", sorted({f[:2] for f in frames}))
