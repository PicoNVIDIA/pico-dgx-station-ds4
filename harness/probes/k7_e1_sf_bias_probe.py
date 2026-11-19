#!/usr/bin/env python3
"""K7 experiment E1 (BOX ONLY. UNTESTED: written on a Mac with no GPU; only py_compile was run on it).

Question (research/K7-packed-scales.md, variant a2): is the block-scaled tensor-core multiply exact when a constant
is moved from the weight scale codes to the activation scale codes? MegaMoE multiplies fp8 activations (one E8M0
scale per 32 elements) by fp4 weights (one E8M0 scale per 32 elements). If the hardware simply adds the two
exponents, then (weight code - k, activation code + k) gives the same bits as (weight code, activation code) for
every k. With k = 112 the weight code is its own low nibble (117..125 -> 5..13), which is what a 4-bit packed scale
bank expands to with no arithmetic. If the outputs differ for any k, variant a2 is dead and only a1 (expand to the
true codes in shared memory) remains.

What it does, with STOCK DeepGEMM and no kernel change, on one real layer:
  part 1 (L1): L1 weight scales - k and the input scales x_sf + k, for k in 1, 8, 64, 112. Wanted: y bit-identical
               to k = 0 at every T and for fast_math on and off.
  part 2 (L2): L2 weight scales - k only, for k in 1, 8. Wanted: y == y0 * 2^-k exactly (a power of two commutes
               with the rounding when nothing underflows). This shows the L2 multiply is linear in the scale code
               too; its activation scale is computed inside the kernel, so the pair shift cannot be tested from
               outside.

It reuses the mini-test harness of the gracetier checkout (bench/gracetier_minitest.py: LayerSetup, mega_call,
set_inputs, quant_rows, synth_x, synth_routing, synth_weights, compare_bits). Copy this file into that checkout's
bench/ directory and run it the way bench/box/gracetier_minitest.sh runs the probe (same container, same mounts),
with the server stopped: it loads one layer's 384 experts (6.6 GiB) into GPU memory.

    python3 bench/k7_e1_sf_bias_probe.py --layer 10 --model-dir /ref --rowmap <rowmap json> [--json scratch/k7-e1.json]

Exit code 0 = every comparison exact, 4 = at least one differs, 2 = could not run.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

KS_L1 = (1, 8, 64, 112)
KS_L2 = (1, 8)
TS = (1, 6, 16)


def shifted(bank, k):
    """int32 MN-major scale bank with k subtracted from each of the 4 code bytes of every word (same strides)."""
    import torch

    lo = int(bank.min())                       # the smallest word bounds the smallest top byte only: check bytes
    b = torch.empty(bank.shape, dtype=torch.int32, device=bank.device).copy_(bank).view(torch.uint8)
    if int(b.min()) < k:
        raise ValueError(f"a scale code {int(b.min())} is smaller than the shift {k} (word min {lo:#x})")
    out = torch.empty_like(bank)
    out.copy_(bank - int(k) * 0x01010101)      # no borrow between bytes: every byte >= k
    if tuple(out.stride()) != tuple(bank.stride()):
        raise RuntimeError(f"shifted bank strides {tuple(out.stride())} != {tuple(bank.stride())}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="K7 E1: scale-code shift exactness of the block-scaled multiply")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--model-dir", default="/ref")
    ap.add_argument("--rowmap", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    import torch

    import gracetier_minitest as MT

    dg = MT._dg()
    setup = MT.LayerSetup(a, dg, want_ref=True, want_stock=False)
    dev = setup.dev
    rng = random.Random(a.seed)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    rows, bad, bad_ctl = [], 0, 0
    try:
        sf1 = {k: shifted(setup.sf1, k) for k in KS_L1}
        sf2 = {k: shifted(setup.sf2, k) for k in KS_L2}
        for T in TS:
            ids = MT.synth_routing(T, 0, setup.hot, setup.cold, MT.TOPK, rng)
            groups = setup.groups_of(torch.tensor(ids, dtype=torch.long, device="cpu"))
            w = torch.tensor(MT.synth_weights(T, MT.TOPK, rng), dtype=torch.float32, device="cpu").to(dev)
            q, sc = MT.quant_rows(MT.synth_x(T, setup.D, g).to(dev))
            top = int(sc.max())
            for fm in (False, True):
                MT.set_inputs(setup.symm, groups, w, q, sc)
                y0 = setup.mega_ref(fm)[:T].clone()
                # Controls (added after the first run passed 36 of 36): prove the comparison CAN fail. The output must
                # be mostly non-zero, and each shift WITHOUT its compensation must change it.
                yf = y0.float()
                stats = {"y0_nonzero_frac": round(float((yf != 0).float().mean()), 4),
                         "y0_abs_max": float(yf.abs().max()), "y0_abs_mean": float(yf.abs().mean())}
                yc1 = MT.mega_call(dg, setup.y, setup.ref_l1, sf1[KS_L1[0]], setup.ref_l2, setup.sf2, setup.symm,
                                   setup.lim, fm)[:T].clone()
                c1 = MT.compare_bits(yc1, y0)
                yc2 = MT.mega_call(dg, setup.y, setup.ref_l1, setup.sf1, setup.ref_l2, sf2[KS_L2[0]], setup.symm,
                                   setup.lim, fm)[:T].clone()
                c2 = MT.compare_bits(yc2, y0)
                ctl_ok = (not c1["equal"]) and (not c2["equal"]) and stats["y0_nonzero_frac"] > 0.5
                bad_ctl += int(not ctl_ok)
                rows.append({"part": "control", "T": T, "fast_math": fm, **stats, "ok": ctl_ok,
                             "l1_unpaired_equal": c1["equal"], "l1_unpaired_frac_bytes_equal": c1["frac_bytes_equal"],
                             "l2_unscaled_equal": c2["equal"], "l2_unscaled_frac_bytes_equal": c2["frac_bytes_equal"]})
                MT.log(f"[E1] control T={T:<2d} fm={int(fm)}: y0 non-zero {stats['y0_nonzero_frac']}, abs max "
                       f"{stats['y0_abs_max']:.4g}, mean {stats['y0_abs_mean']:.4g}; unpaired L1 shift changes y: "
                       f"{not c1['equal']} (bytes equal {c1['frac_bytes_equal']}); L2 shift changes y: "
                       f"{not c2['equal']} (bytes equal {c2['frac_bytes_equal']}); control ok {ctl_ok}")
                for k in KS_L1:                                                  # part 1: the pair shift on L1
                    if top + k > 254:
                        rows.append({"part": "L1", "T": T, "fast_math": fm, "k": k,
                                     "skipped": f"activation code {top} + {k} > 254"})
                        continue
                    MT.set_inputs(setup.symm, groups, w, q, (sc.to(torch.int32) + k).to(torch.uint8))
                    y1 = MT.mega_call(dg, setup.y, setup.ref_l1, sf1[k], setup.ref_l2, setup.sf2, setup.symm,
                                      setup.lim, fm)[:T].clone()
                    r = MT.compare_bits(y1, y0)
                    bad += int(not r["equal"])
                    rows.append({"part": "L1", "T": T, "fast_math": fm, "k": k, "max_act_code": top, **r})
                    MT.log(f"[E1] L1 pair shift k={k:<3d} T={T:<2d} fm={int(fm)}: equal {r['equal']} (rows exact "
                           f"{r['rows_exact']}/{T}, frac {r['frac_bytes_equal']}, max abs {r['max_abs']})")
                for k in KS_L2:                                                  # part 2: L2 linear in the code
                    MT.set_inputs(setup.symm, groups, w, q, sc)
                    y1 = MT.mega_call(dg, setup.y, setup.ref_l1, setup.sf1, setup.ref_l2, sf2[k], setup.symm,
                                      setup.lim, fm)[:T].clone()
                    want = (y0.float() * (2.0 ** -k)).to(torch.bfloat16)
                    r = MT.compare_bits(y1, want)
                    bad += int(not r["equal"])
                    rows.append({"part": "L2", "T": T, "fast_math": fm, "k": k, **r})
                    MT.log(f"[E1] L2 weight shift k={k:<3d} T={T:<2d} fm={int(fm)}: y == y0 * 2^-{k}: {r['equal']} "
                           f"(frac {r['frac_bytes_equal']}, max abs {r['max_abs']})")
    finally:
        setup.close()
    verdict = "EXACT: variant a2 is alive" if bad == 0 else f"{bad} comparison(s) differ: variant a2 is dead"
    if bad_ctl:
        verdict = f"CONTROLS FAILED ({bad_ctl}): this probe cannot tell a shift from no shift; ignore its result"
    MT.log(f"[E1] {verdict} (layer {a.layer}, {len(rows)} comparisons)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"schema": "k7-e1/2", "layer": a.layer, "verdict": verdict, "rows": rows}, f, indent=1)
    return 3 if bad_ctl else (0 if bad == 0 else 4)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - one line for the driver, then the trace
        print(f"[E1] could not run: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(2)
