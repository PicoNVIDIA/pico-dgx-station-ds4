#!/usr/bin/env python3
"""Mac-side test of k7_e2_probe.py's logic (NO GPU, NO compiler): the launches are answered by a small pure-Python
model of the kernel, once for every outcome the decision table of research/K7-E2-design.md lists. It checks that
each modelled outcome is classified and decided as intended, and that the failure paths give the right exit code.

It does NOT test the kernel, the PTX, the tensor-map encoding or anything the GPU does. The model's behaviours are
hypotheses (one of them is what NVIDIA's manuals describe); the real behaviour is what E2 measures on the box.

    python3 t_k7_e2_sim.py        (exit 0 = all checks pass)
"""
import contextlib
import io
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k7_e2_probe as K  # noqa: E402


def e4m3(b):
    s, e, m = (-1.0 if b & 0x80 else 1.0), (b >> 3) & 15, b & 7
    return s * ((m / 8.0) * 2.0 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7))


def f32(x):
    try:
        return struct.pack("<f", x)
    except OverflowError:
        return struct.pack("<f", float("inf") if x > 0 else float("-inf"))


class Sim:
    """A model of one launch. tma4: what a 16U4_ALIGN16B load leaves ('packpad' as the manuals say, 'bytes' = one
    byte per nibble). cp4 / cp6: what the decompressing copy makes of a value ('zext', 'shift2', 'code')."""

    def __init__(self, tma4="packpad", cp4="shift2", cp6="zext", order4=0, order6=0, pad="keep", tx="packed",
                 bad_ctl=False, refuse=(), refuse_plain_map=(), dead=False, fault_forms=(), bad_mma=False):
        self.__dict__.update(locals())
        self.calls = 0

    def __call__(self, req):
        self.calls += 1
        if self.dead:
            return {"rc": None, "exception": "OSError: libcuda.so.1: cannot open shared object file"}
        form = req["form"]
        if form in self.fault_forms:
            return {"rc": 5, "cuda_err": 715, "encode_err": 0, "err": "an illegal instruction was encountered"}
        image = bytes.fromhex(req["image"])
        smem = bytearray(image[K.SF_OFF:K.SF_OFF + K.SF_BYTES])
        status = [K.NOTW * 0x01010101] * 16
        if req["src_mode"] == 1:
            kind, rb = req["kind"], K.ROW_BYTES[req["kind"]]
            if kind in self.refuse or (kind in self.refuse_plain_map and not req["swizzle"]):
                return {"rc": 2, "cuda_err": 0, "encode_err": 1, "err": ""}
            src = bytes.fromhex(req["src"])
            box = src[req["c1"] * rb:(req["c1"] + req["box1"]) * rb]
            true_tx = len(box)
            chunks = []                                   # (16-byte row index, bytes written at its start)
            if kind in ("u8", "4a8"):
                land = box
                chunks = [(i // 16, land[i:i + 16]) for i in range(0, len(land), 16)]
            elif kind == "4a16" and self.tma4 in ("bytes", "bytes_swapped"):
                q = [(box[i // 2] >> (4 * (i % 2))) & 15 for i in range(2 * len(box))]
                if self.tma4 == "bytes_swapped":
                    q = [q[i ^ 1] for i in range(len(q))]
                chunks = [(i // 16, bytes(q[i:i + 16])) for i in range(0, len(q), 16)]
            else:
                db = K.DATA_BYTES[kind]
                for r in range(len(box) // db):
                    row = box[db * r:db * (r + 1)]
                    chunks.append((r, row + (bytes(16 - db) if self.pad == "zero" else b"")))
                if self.tx == "footprint":
                    true_tx = 16 * (len(box) // db)
            for r, data in chunks:
                pos = 16 * r
                if req["swizzle"]:
                    br, j = r // 8, r % 8
                    pos = br * 128 + ((j ^ (br & 7)) << 4)
                smem[pos:pos + len(data)] = data
            status[0], status[1] = (1 if req["expect_tx"] == true_tx else 0), 1
        rows = K.FORM_ROWS[form]

        def conv(row):
            if form in (0, 5):
                return bytes(row)
            if form in (1, 3):
                v = [(row[j // 2] >> (4 * (j % 2))) & 15 for j in range(16)]
                how, order = self.cp4, self.order4
            else:
                x = int.from_bytes(bytes(row[:12]), "little")
                v = [(x >> (6 * j)) & 63 for j in range(16)]
                how, order = self.cp6, self.order6
            if order == 1:
                v = [v[j ^ 1] for j in range(16)]
            f = {"zext": lambda t: t, "shift2": lambda t: (t << 2) & 0xFF, "code": lambda t: 0x70 | t}[how]
            return bytes(f(t) for t in v)

        sf = [conv(smem[16 * (i % rows):16 * (i % rows) + 16]) for i in range(128)]
        tmem = b"".join(x + bytes([K.SENT]) * 16 for x in sf)
        ctl = b"".join(struct.pack("<8I", *[(0xC0DE0000 | (r << 8) | c) ^ (1 if self.bad_ctl and r == 77 else 0)
                                           for c in range(8)]) for r in range(128))
        acc = bytes([K.NOTW]) * (128 * 16 * 4)
        status[2], status[3], status[6], status[7] = 1, 1, 0, K.DONE_MAGIC
        if req["do_mma"]:
            a = [[e4m3(image[K.A_OFF + K.sw128_offset(m, k)]) for k in range(128)] for m in range(128)]
            b = [[e4m3(image[K.B_OFF + K.sw128_offset(n, k)]) for k in range(128)] for n in range(16)]
            out = bytearray()
            for m in range(128):
                for n in range(16):
                    x = 0.0
                    for blk in range(4):
                        dot = sum(a[m][32 * blk + k] * b[n][32 * blk + k] for k in range(32))
                        ea = sf[m][(m // 32) * 4 + blk] - 127
                        eb = image[K.SFB_OFF + n * 16 + blk] - 127
                        x += dot * 2.0 ** (ea + eb)
                    out += f32(x * (3.0 if self.bad_mma else 1.0))
            acc = bytes(out)
            status[4], status[5] = 1, 1
        return {"rc": 0, "cuda_err": 0, "encode_err": 0, "err": "", "status": status, "ctl": ctl.hex(),
                "smem": bytes(smem).hex(), "tmem": tmem.hex(), "acc": acc.hex(),
                "device": "simulated (no GPU)", "cp_string": K.FORM_STR[form]}


FAILS = []
COUNT = [0]


def check(name, cond, detail=""):
    COUNT[0] += 1
    if not cond:
        FAILS.append(name)
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or not detail else "  <- " + detail))


def run(sim, forms=(0, 1, 2), skip_mma=False):
    ctx = K.Ctx(sim, {f: f"sim-form-{f}" for f in forms}, skip_mma=skip_mma)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        S = K.experiment(ctx)
    return S, ctx, buf.getvalue()


def has(ctx, *subs):
    return any(all(s in v for s in subs) for v in ctx.verdicts)


def ends(ctx, start, end):
    """A verdict that starts with `start` and ends with `end` (so 'byte = value' cannot match 'byte = value << 2')."""
    return any(v.startswith(start) and v.endswith(end) for v in ctx.verdicts)


T4 = "TMA 4-bit ALIGN16B > plain cp, all values"


def main():
    # ---- unit checks of the helpers
    q = K.nib_all(512)
    check("nib_all: every nibble once per row", all(sorted(q[16 * r:16 * r + 16]) == list(range(16)) for r in range(32)))
    check("nib_real: only 5..15, all 11 values", sorted(set(K.nib_real(512))) == list(range(5, 16)))
    check("six_all: all 64 values, 16 distinct per row", sorted(set(K.six_all(512))) == list(range(64)) and
          all(len(set(K.six_all(512)[16 * r:16 * r + 16])) == 16 for r in range(32)))
    check("pack4 is the K7 packer order (even index in the low nibble)", K.pack4([1, 2, 3, 4]) == bytes([0x21, 0x43]))
    v = K.six_all(32)
    x = int.from_bytes(K.pack6(v)[:12], "little")
    check("pack6 round trip", [(x >> (6 * j)) & 63 for j in range(16)] == v[:16] and len(K.pack6(v)) == 24)
    check("padded rows", K.padded(bytes(range(16)), 8, 0x5A) == bytes(range(8)) + b"\x5a" * 8 + bytes(range(8, 16)) + b"\x5a" * 8)
    sw = sorted(K.sw128_offset(r, k) for r in range(128) for k in range(128))
    check("sw128_offset is a bijection onto the tile", sw == list(range(16384)))
    check("oracle_ones is exact and has the right size", len(K.oracle_ones([0x70 | n for n in K.nib_real(512)])) == 8192)
    rows = K.rows16(bytes(q), 32)
    fm = K.find_map([[t << 2 for t in r] for r in rows], rows)
    check("find_map: value << 2 in order", fm is not None and fm[0] == 0 and K.map_name(fm[1])[0] == "SHIFTED<<2")
    fm = K.find_map([[r[j ^ 1] for j in range(16)] for r in rows], rows)
    check("find_map: neighbours swapped, zero-extended", fm is not None and fm[0] == 1 and
          K.map_name(fm[1])[0] == "ZERO-EXTENDED")
    check("find_map: a constant output is not a map", K.find_map([[0xEE] * 16] * 32, rows) is None)
    check("find_map: packed bytes are not a map", K.find_map(K.rows16(K.pack4(q) + bytes(256), 32), rows) is None)

    # ---- 1. what NVIDIA's manuals describe: packed + pad in shared memory, value << 2 after the 4-bit copy,
    #         zero-extended after the 6-bit copy
    S, ctx, text = run(Sim())
    check("manuals: exit 0", S["exit_code"] == 0, str(S["exit_code"]))
    check("manuals: controls pass", S["ctl_manual"] and S["ctl_tma"])
    check("manuals: 4-bit load is still packed, 256 bytes", has(ctx, f"smem [{T4}]", "STILL PACKED, 8 data + 8 pad",
                                                                 "pad untouched", "barrier counts 256 bytes"))
    check("manuals: ALIGN8B is packed without gaps", has(ctx, "smem [TMA 4-bit ALIGN8B", "STILL PACKED, no gaps"))
    check("manuals: plain copy of the packed load is a raw copy", has(ctx, f"tmem [{T4}]", "RAW COPY"))
    check("manuals: 4-bit copy alone = value << 2", has(ctx, "tmem [padded rows > 4-bit cp, all values]",
                                                        "in order", "byte = value << 2"))
    check("manuals: 4-bit route = value << 2", has(ctx, "tmem [TMA 4-bit ALIGN16B > 4-bit cp, nibbles 5..15]",
                                                   "byte = value << 2"))
    check("manuals: 6-bit copy alone zero-extended", ends(ctx, "tmem [padded rows > 6-bit cp, all values]", "byte = value"))
    check("manuals: 6-bit route zero-extended", ends(ctx, "tmem [TMA 6-bit ALIGN16B > 6-bit cp, codes-64]", "byte = value"))
    check("manuals: verdict lines are short", max(len(v) for v in ctx.verdicts) <= 190, str(max(len(v) for v in ctx.verdicts)))
    check("manuals: multiply harness passes", S.get("mma_harness") is True)
    check("manuals: 4-bit multiply differs", S["mma"]["4bit-decomp"]["rand_equal"] is False)
    check("manuals: 6-bit multiply identical", S["mma"]["6bit-decomp"]["rand_equal"] and S["mma"]["6bit-decomp"]["ones_exact"])
    check("manuals: decision = 6-bit alive", S["decision"].startswith("4-BIT A2 IS DEAD; A2 WITH 6-BIT CODES IS ALIVE"),
          S["decision"])
    check("manuals: every dump is at most 64 bytes wide", all(len(ln) < 140 for ln in text.splitlines() if ": " in ln and
                                                               ln.strip().startswith("[E2]        0")))

    # ---- 2. the hoped-for outcome: the 4-bit copy zero-extends
    S, ctx, _ = run(Sim(cp4="zext"))
    check("zext: decision = a2 as designed (decompressing copy)", S["decision"] == "A2 AS DESIGNED (decompressing copy)",
          S["decision"])
    check("zext: multiply confirms", S["mma"]["4bit-decomp"]["rand_equal"] and S["exit_code"] == 0)
    check("zext: note says confirmed", any("confirms it" in w for w in S["why"]))

    # ---- 3. zero-extended but the neighbours swapped
    S, ctx, _ = run(Sim(cp4="zext", order4=1))
    check("swapped: decision", S["decision"] == "A2 WITH THE NIBBLES SWAPPED AT PACK TIME (decompressing copy)",
          S["decision"])
    check("swapped: the multiply ran with the swapped packing and confirms", S["mma"]["4bit-decomp"]["rand_equal"] and
          has(ctx, "multiply [4-bit decompress, act +112, neighbours swapped at pack", "IDENTICAL") and
          has(ctx, "tmem [TMA 4-bit ALIGN16B > 4-bit cp, all values]", "NEIGHBOURS SWAPPED"))

    # ---- 4. the load itself gives one byte per nibble: plain copy is enough
    S, ctx, _ = run(Sim(tma4="bytes"))
    check("bytes: shared memory verdict", has(ctx, f"smem [{T4}]", "ONE BYTE PER ELEMENT, in order", "byte = value; "))
    check("bytes: decision = a2 as designed (plain copy)", S["decision"] == "A2 AS DESIGNED (plain copy)", S["decision"])
    check("bytes: multiply via the plain copy confirms", S["mma"]["4bit-plain"]["rand_equal"])
    S, ctx, _ = run(Sim(tma4="bytes_swapped"))
    check("bytes swapped: decision", S["decision"] == "A2 WITH THE NIBBLES SWAPPED AT PACK TIME (plain copy)", S["decision"])
    check("bytes swapped: multiply confirms with the swapped packing", S["mma"]["4bit-plain"]["rand_equal"])

    # ---- 5. nothing usable: both copies shift
    S, ctx, _ = run(Sim(cp6="shift2"))
    check("both shift: decision = a1 or a0", S["decision"].startswith("NEEDS A1 OR A0"), S["decision"])
    check("both shift: exit 0 (it ran and classified)", S["exit_code"] == 0)

    # ---- 6. the assembler refused the decompressing forms (only the control compiled)
    S, ctx, _ = run(Sim(), forms=(0,))
    check("no forms: decision = a1 or a0", S["decision"].startswith("NEEDS A1 OR A0"), S["decision"])
    check("no forms: says so", has(ctx, "decompressing copy (4-bit): NOT AVAILABLE") and
          any("no decompressing copy assembled" in w for w in S["why"]))

    # ---- 7. form 1 refused, the 128x128b fallback shape works
    S, ctx, _ = run(Sim(cp4="zext"), forms=(0, 2, 3, 5))
    check("fallback shape: classified through form 3", ends(ctx, "tmem [padded rows > 4-bit cp, all values]",
                                                            "byte = value"))
    check("fallback shape: not sold as a2 (layout cannot use it); 6-bit decides", "6-BIT" in S["decision"], S["decision"])
    check("fallback shape: its own plain control ran", S.get("ctl_128") is True)

    # ---- 8. controls fail
    S, ctx, _ = run(Sim(bad_ctl=True))
    check("bad st/ld control: NO DECISION, exit 3", S["decision"].startswith("NO DECISION") and S["exit_code"] == 3,
          f"{S['decision']} / {S['exit_code']}")
    S, ctx, _ = run(Sim(refuse=("u8",)))
    check("TMA control cannot run: exit 3, flagged", S["exit_code"] == 3 and not S["ctl_tma"] and
          any("TMA control failed" in w for w in S["why"]), f"{S['decision']} / {S['exit_code']}")

    # ---- 9. nothing runs at all
    S, ctx, text = run(Sim(dead=True))
    check("no GPU / library: exit 2 and one clear line", S["exit_code"] == 2 and "could not run" in text and
          has(ctx, "COULD NOT RUN", "libcuda"), text[-300:])

    # ---- 10. the decompressing kernel faults at run time: the rest still runs
    S, ctx, _ = run(Sim(fault_forms=(1,)))
    check("form 1 faults: reported, 6-bit still decides", has(ctx, "COULD NOT RUN", "illegal instruction") and
          "6-BIT" in S["decision"] and S["exit_code"] == 0, S["decision"])

    # ---- 11. the barrier counts the shared-memory footprint, not the packed bytes
    S, ctx, _ = run(Sim(tx="footprint"))
    check("tx discovery: 512 bytes found", has(ctx, f"smem [{T4}]", "barrier counts 512 bytes"))

    # ---- 12. the plain (no swizzle) map is refused for the 4-bit type: SWIZZLE_128B fallback, un-swizzled dump
    S, ctx, text = run(Sim(refuse_plain_map=("4a16",)))
    check("swizzle fallback: still classified as packed + pad", has(ctx, f"smem [{T4}]", "STILL PACKED, 8 data + 8 pad")
          and "SWIZZLE_128B map" in text)

    # ---- 13. the load zeroes the pad instead of leaving it
    S, ctx, _ = run(Sim(pad="zero"))
    check("pad zero: reported", has(ctx, f"smem [{T4}]", "pad zero"))

    # ---- 14. dumps say a2, the multiply says no: no decision
    S, ctx, _ = run(Sim(cp4="zext", bad_mma=True))
    check("harness broken: decision rests on the dumps, says so", S["decision"] == "A2 AS DESIGNED (decompressing copy)"
          and any("rests on the dumps alone" in w for w in S["why"]) and S.get("mma_harness") is False, S["decision"])

    # ---- 15. --skip-mma
    S, ctx, _ = run(Sim(cp4="zext"), skip_mma=True)
    check("skip-mma: decision without stage 3", S["decision"] == "A2 AS DESIGNED (decompressing copy)" and "mma" not in S)

    # ---- 16. the 4-bit copy returns the true code (0x70 | n = n + 112): a2 with nothing left for the activations
    S, ctx, _ = run(Sim(cp4="code"))
    check("code map: a2, activations need +0, multiply confirms", has(ctx, "byte = value +112") and
          S["decision"] == "A2 AS DESIGNED (decompressing copy)" and any("need +0" in w for w in S["why"]) and
          S["mma"]["4bit-decomp"]["rand_equal"], S["decision"] + " | " + " | ".join(S["why"]))

    print(f"{COUNT[0] - len(FAILS)} of {COUNT[0]} checks pass")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
