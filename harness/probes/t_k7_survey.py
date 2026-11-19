#!/usr/bin/env python3
"""Local test of k7_scale_survey.py on synthetic checkpoints with the real expert shapes. No GPU.

    env -u PYTHONPATH python3 t_k7_survey.py                       (numpy only: parts 1, 2, 4, 5)
    uv run --no-project --with torch --with numpy python t_k7_survey.py <gt-built dir>
                                                                   (adds part 3: the bank layout against
                                                                    tree/fastpath/gracetier_rt/runtime/dg_mega.py)
"""
import json
import os
import shutil
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k7_scale_survey as S  # noqa: E402

I, D = 2304, 5120
RNG = np.random.default_rng(7)
CODES = np.arange(117, 126, dtype=np.uint8)
P = np.array([0.002, 0.01, 0.06, 0.25, 0.42, 0.2, 0.05, 0.006, 0.002])
P = P / P.sum()
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def rand_codes(shape):
    return RNG.choice(CODES, size=shape, p=P).astype(np.uint8)


def write_shards(root, tensors, nshards=2):
    """tensors: {name: uint8 array}. Splits them over nshards safetensors files (header + raw bytes)."""
    names = list(tensors)
    for s in range(nshards):
        part = names[s::nshards]
        hdr, off, blobs = {"__metadata__": {"format": "pt"}}, 0, []
        for n in part:
            b = tensors[n].tobytes()
            hdr[n] = {"dtype": "U8", "shape": list(tensors[n].shape), "data_offsets": [off, off + len(b)]}
            off += len(b)
            blobs.append(b)
        h = json.dumps(hdr).encode()
        h += b" " * (-len(h) % 8)
        with open(os.path.join(root, f"model-{s + 1:05d}-of-{nshards:05d}.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            for b in blobs:
                f.write(b)


def make_ckpt(root, layers, n_exp, mutate=None):
    t = {}
    for l in layers:
        for e in range(n_exp):
            t[f"layers.{l}.ffn.experts.{e}.w1.scale"] = rand_codes((I, D // 32))
            t[f"layers.{l}.ffn.experts.{e}.w3.scale"] = rand_codes((I, D // 32))
            t[f"layers.{l}.ffn.experts.{e}.w2.scale"] = rand_codes((D, I // 32))
            t[f"layers.{l}.ffn.experts.{e}.w1.weight"] = np.zeros((2, 3), np.uint8)       # must be ignored
    t["layers.0.attn.wq_a.scale"] = np.zeros((4, 4), np.uint8)                              # must be ignored
    if mutate:
        mutate(t)
    write_shards(root, t)
    return t


def run(root, *extra):
    out = os.path.join(root, "out.json")
    rc = S.main([root, "--json", out] + list(extra))
    with open(out) as f:
        return rc, json.load(f)


def utccp_pos(n):
    a, o = divmod(int(n), 128)
    return a * 128 + (o % 32) * 4 + o // 32


def l1_row(r, p):
    return (r // 8) * 16 + p * 8 + r % 8


def main():
    tmp = tempfile.mkdtemp(prefix="k7survey-")
    try:
        # ---------------------------------------------------------------- 1. a clean checkpoint with planted cases
        def mut(t):
            for p in ("w1", "w3", "w2"):                                      # expert 3 == expert 2 in layer 0
                t[f"layers.0.ffn.experts.3.{p}.scale"] = t[f"layers.0.ffn.experts.2.{p}.scale"].copy()
            t["layers.1.ffn.experts.5.w2.scale"][:] = 120                     # one constant L2 block
        a = os.path.join(tmp, "a")
        os.mkdir(a)
        truth = make_ckpt(a, (0, 1), 8, mut)
        rc, r = run(a)
        allb = np.concatenate([v.ravel() for k, v in truth.items() if k.endswith(".scale") and ".experts." in k])
        check("1 exit code", rc == 0)
        check("1 every scale byte read", r["codes_read"] == allb.size == 2 * 8 * 1105920, str(r["codes_read"]))
        check("1 distinct values", r["distinct"] == sorted(set(allb.tolist())), str(r["distinct"]))
        check("1 none outside 0x70..0x7F", r["outside_0x70_0x7F"] == 0 and r["layers_span_le_15"])
        h = np.bincount(allb, minlength=256).astype(float)
        p = h[h > 0] / h.sum()
        check("1 order-0 entropy", abs(r["H0"] + float((p * np.log2(p)).sum())) < 1e-9, f"{r['H0']:.4f}")
        check("1 whole-expert duplicate found (L1 and L2)", r["expert_dups"]["l1"]["dup_model"] == 1
              and r["expert_dups"]["l2"]["dup_model"] == 1, str(r["expert_dups"]))
        check("1 2 MiB granules: none duplicate on random scales", r["pages"][str(2 << 20)]["dup_model"] == 0,
              str(r["pages"][str(2 << 20)]))
        c4 = r["pages"][str(4 << 10)]
        check("1 constant 4 KiB chunks seen for the constant block", c4["constant"] >= 368640 // 4096 - 2
              and c4["dup_model"] >= c4["constant"] - 1, str(c4))
        check("1 pack self-check ran and is clean", r["packcheck"]["bytes"] > 0 and r["packcheck"]["mismatch"] == 0,
              str(r["packcheck"]))
        n2 = r["pages"][str(2 << 20)]["chunks"]
        want = 2 * (8 * 737280 // (2 << 20) + 8 * 368640 // (2 << 20))
        check("1 granule count", n2 == want, f"{n2} vs {want}")

        # ---------------------------------------------------------------- 2. rowmap order changes pages, not codes
        rm = {"layers": {str(l): {"hot": [7, 0, 3, 5, 1], "cold": [2, 6, 4]} for l in (0, 1)}}
        with open(os.path.join(a, "rm.json"), "w") as f:
            json.dump(rm, f)
        rc, r2 = run(a, "--rowmap", os.path.join(a, "rm.json"))
        check("2 rowmap run: same codes", r2["distinct"] == r["distinct"] and r2["codes_read"] == r["codes_read"]
              and abs(r2["H0"] - r["H0"]) < 1e-12)
        check("2 rowmap run: expert duplicates unchanged", r2["expert_dups"] == r["expert_dups"])
        rc, r3 = run(a, "--no-pages", "--layers", "1")
        check("2 --no-pages --layers 1", rc == 0 and r3["pages"] == {} and r3["codes_read"] == 8 * 1105920)

        # ---------------------------------------------------------------- 3. bank layout against dg_mega.py
        c1, c3, c2 = rand_codes((3, I, 160)), rand_codes((3, I, 160)), rand_codes((3, D, 72))
        b1, b2 = S.bank_l1(c1, c3), S.bank_l2(c2)
        ok = True
        for _ in range(4000):                                               # scalar formulas from dg_mega's docstrings
            g, r_, p_, cw = RNG.integers(3), RNG.integers(I), RNG.integers(2), RNG.integers(40)
            src = (c1, c3)[p_][g, r_, 4 * cw:4 * cw + 4].view("<i4")[0]
            ok &= b1[g, cw, utccp_pos(l1_row(int(r_), int(p_)))] == src
            r_, cw = RNG.integers(D), RNG.integers(18)
            ok &= b2[g, cw, utccp_pos(int(r_))] == c2[g, r_, 4 * cw:4 * cw + 4].view("<i4")[0]
        check("3 bank layout == the index formulas (l1_row, utccp_pos, MN-major)", bool(ok))
        check("3 bank sizes", b1.nbytes == 3 * 737280 and b2.nbytes == 3 * 368640)
        gt = sys.argv[1] if len(sys.argv) > 1 else None
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or gt is None:
            print("SKIP 3 bank layout against runtime/dg_mega.py (needs torch and the gt-built dir as argv[1])")
        else:
            sys.path.insert(0, os.path.join(gt, "tree", "fastpath", "gracetier_rt"))
            from runtime import dg_mega as DM
            t1 = DM.mega_l1_sf(torch.from_numpy(c1), torch.from_numpy(c3), "mn")
            t2 = DM.mega_l2_sf(torch.from_numpy(c2), "mn")
            check("3 dg_mega strides are MN-major", tuple(t1.stride()) == (4608 * 40, 1, 4608)
                  and tuple(t2.stride()) == (5120 * 18, 1, 5120), f"{tuple(t1.stride())} {tuple(t2.stride())}")
            s1 = t1.transpose(-1, -2).contiguous().numpy()
            s2 = t2.transpose(-1, -2).contiguous().numpy()
            check("3 L1 bank bytes == dg_mega.mega_l1_sf storage", np.array_equal(s1, b1))
            check("3 L2 bank bytes == dg_mega.mega_l2_sf storage", np.array_equal(s2, b2))

        # ---------------------------------------------------------------- 4. a code outside the nibble range
        def mut_bad(t):
            t["layers.0.ffn.experts.1.w3.scale"][17, 3] = 0x6F
        b = os.path.join(tmp, "b")
        os.mkdir(b)
        make_ckpt(b, (0,), 4, mut_bad)
        rc, r4 = run(b)
        check("4 one code outside 0x70..0x7F is counted", r4["outside_0x70_0x7F"] == 1 and r4["min"] == 0x6F)
        check("4 span still <= 15 (a base + nibble pack would work)", r4["layers_span_le_15"])
        check("4 pack self-check skipped for that layer", r4["packcheck"]["skipped_layers"] == 1)
        check("4 table says NOT valid", "NOT valid" in S.table(r4))

        # ---------------------------------------------------------------- 5. positive control for page duplicates
        def mut_const(t):
            for k in t:
                if ".experts." in k and k.endswith(".scale"):
                    t[k][:] = 0x78
        c = os.path.join(tmp, "c")
        os.mkdir(c)
        make_ckpt(c, (0,), 6, mut_const)
        rc, r5 = run(c)
        p2 = r5["pages"][str(2 << 20)]
        check("5 constant scales: every 2 MiB granule is a duplicate of one", p2["chunks"] == 3
              and p2["dup_model"] == p2["chunks"] - 1 and p2["constant"] == p2["chunks"], str(p2))
        check("5 entropy 0, 1 distinct value", r5["H0"] == 0.0 and r5["distinct"] == [0x78])

        # ---------------------------------------------------------------- pack round trip on random bytes
        x = rand_codes((5, 18, 20480))
        check("pack/unpack round trip", np.array_equal(S.unpack_nibbles(S.pack_nibbles(x)), x)
              and S.pack_nibbles(x).nbytes * 2 == x.nbytes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
