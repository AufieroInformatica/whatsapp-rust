#!/usr/bin/env python3
"""Reduced-rate vectors from the shipped VoIP module.

`spec` turns an existing packet-by-packet derivation spec into one that sets the encoder's bitrate
(`OPUS_SET_BITRATE`, 4002, through the module's own `encoder_ctl`) right after `opus_open`, and drops
the self-check hashes that describe the original full-rate outputs. `fixture` packs what the
derivation wrote into the `lr_wasm_*` testdata files.

    python3 wasm_lowrate.py spec tools/oracle-core/specs/mlow_dtx_off.json OUT.json 8000 \
        tools/oracle-core/specs/mlow_dtx_off.json [LOSS_PERCENT]
    oracle derive --spec OUT.json -o DIR          # `oracle` from tools/oracle-cli, release build
    python3 wasm_lowrate.py fixture DIR packet decoded raw FRAMES.json REF.raw

For the 120 ms spec the packet files are `pkt120_N.bin` and `dec120_N.raw` instead (the names the
spec uses). Set the bitrate at 8000-8500 for 60/120 ms: the module drops to the reduced rate at or
under about 8.7 kbps.
"""
import json, os, struct, sys


def spec(src, dst, bitrate, donor, loss_percent=0):
    d = json.load(open(src))
    if "encoder_ctl" not in d["functions"]:
        d["functions"]["encoder_ctl"] = json.load(open(donor))["functions"]["encoder_ctl"]
    steps = d["steps"]
    steps[:] = [s for s in steps if s.get("op") != "assert_sha256"]
    i = next(k for k, s in enumerate(steps) if s.get("op") == "assert_reg" and s.get("reg") == "$rc")
    steps[i + 1:i + 1] = [
        {"op": "read_u32", "ptr": "$state", "at": 120, "as": "enc_lr"},
        {"op": "malloc", "as": "ctl_arg_lr", "len": 4},
        {"op": "write", "ptr": "$ctl_arg_lr", "hex": struct.pack("<i", bitrate).hex()},
        {"op": "call_table", "func": "encoder_ctl", "args": ["$enc_lr", 4002, "$ctl_arg_lr"],
         "results": ["brrc"]},
        {"op": "assert_reg", "reg": "$brrc", "equals": 0},
    ]
    if loss_percent:
        # In-band FEC (4012) plus the expected loss (4014). The bitrate is then main plus
        # redundancy, so pass about twice the reduced-rate threshold for the main part to fit.
        for request, value in ((4012, 1), (4014, loss_percent)):
            steps[i + 6:i + 6] = [
                {"op": "write", "ptr": "$ctl_arg_lr", "hex": struct.pack("<i", value).hex()},
                {"op": "call_table", "func": "encoder_ctl", "args": ["$enc_lr", request, "$ctl_arg_lr"],
                 "results": ["ctlrc_lr"]},
            ]
            i += 2
    json.dump(d, open(dst, "w"), indent=1)


def fixture(directory, pkt_prefix, dec_prefix, dec_ext, frames_out, pcm_out):
    n = len([f for f in os.listdir(directory) if f.startswith(pkt_prefix)])
    frames, pcm = [], bytearray()
    for i in range(n):
        width = "03d" if pkt_prefix == "packet" else ""
        frames.append(open(f"{directory}/{pkt_prefix}{i:{width}}.bin", "rb").read().hex())
        pcm += open(f"{directory}/{dec_prefix}{i:{width}}.{dec_ext}", "rb").read()
    with open(frames_out, "w") as f:
        json.dump(frames, f, separators=(",", ":"))
        f.write("\n")
    with open(pcm_out, "wb") as f:
        f.write(bytes(pcm))
    print(n, "packets", sorted({h[:2] for h in frames}))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "spec":
        spec(args[0], args[1], int(args[2]), args[3], int(args[4]) if len(args) > 4 else 0)
    elif cmd == "fixture":
        fixture(*args)
    else:
        sys.exit(__doc__)
