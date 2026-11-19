#!/usr/bin/env python3
"""k5_step_table.py: per-kernel table of one decode step. Stdlib only.

Input (auto-detected by file name):
  *.sqlite            nsys export (tables CUPTI_ACTIVITY_KIND_KERNEL + StringIds). This is the proven path.
  *.json / *.json.gz  torch-profiler (Kineto) chrome trace: events with cat == "kernel".

Usage:
  python3 k5_step_table.py TRACE [--delim NAME] [--top N] [--csv out.csv] [--json out.json]
                                 [--main-moe SUBSTR] [--min-steps N]

A step = the kernels between two launches of the delimiter kernel, which fires once per speculative step
(default _rejection_kernel, the same default as Meadlock's w0/step_profile_analyze.py). If it is missing the
script infers one the way his script does (a kernel whose count divides the MegaMoE count).
First and last step are dropped (partial).

Output: step wall (median), GPU-busy union, sum of kernel durations (gap test), duration histogram,
group totals, then one row per kernel: group, calls/step, us/step, avg us, share of step.

Groups: attention, moe_hot, gt_stage, engram, drafter, head, sampler, other.
Names alone cannot tell the drafter's MoE/attention/GEMM launches from the main model's (same kernels).
So each step is also split by position: the "main span" runs from just before the first gt_stage_kernel to the
last MegaMoE that follows a gt_stage_kernel. Shared-name kernels outside the main span are booked to drafter.
Pass --no-span to turn that off. Group rules are a best guess from Meadlock's A2 kernel names
(gracetier docs/gt1/a2/gt1-a2-analyze-20261109-144249.log); check the "other" rows and extend RULES.
"""
import collections
import gzip
import json
import re
import sqlite3
import statistics as st
import sys

# (group, regex on "short name | demangled name"), first match wins.
RULES = [
    ("gt_stage", r"gt_stage"),
    ("sampler", r"_rejection_kernel|rejection|topp|top_p|gumbel|multinomial|_compute_local_logits_stats|argmax|logits_stats"),
    ("drafter", r"markov|draft|dflash|_match_chain|suffix|mega_mix_kernel|mega_seam_kernel|ctxkv|proj_side"),
    ("engram", r"engram"),
    ("head", r"nvjet_sm103_tst_128x8|lm_head|qhead|dghead"),
    ("moe_hot", r"mega_moe|router_kernel|combine"),
    # stage_kernel is Garner's w3 pre-attention stage (kernels/w3-preattn/include/w3/standalone.cuh:39), not MoE.
    # fp8_fp4_gemm_1d1d + MXFP8Quantize (43/step each in Meadlock's A2) look like the stock wo_a einsum + quantize
    # that run because w8oproj is off in our component list. Inference from counts: check their position in the trace.
    ("attention", r"attn|paged_mqa|LowLatencyBlockscaled|low_latency_blockscaled|_mask_candidates|_candidate_flags"
                  r"|topk_global|_indexer|_fused_save_compress|_rope_quant|_compute_global_topk|swa_meta|woa_kernel|rope"
                  r"|(^|[^a-z_])stage_kernel|fp8_fp4_gemm|MXFP8Quantize|mxfp8_quantize|splitKreduce|Kernel2|nvjet_sm103_tss"),
    # mHC mixes and seams sit between attention and MoE. They are their own sub-family; booked to "other".
    ("other", r"mix_kernel|seam_kernel|mhc|hc_pre|rms_norm"),
]
RULES = [(g, re.compile(rx)) for g, rx in RULES]
SHARED = re.compile(r"mega_moe|fp8_fp4_gemm|stage_kernel|MXFP8Quantize|mxfp8_quantize|LowLatencyBlockscaled|"
                    r"low_latency_blockscaled|dense_blockscaled|mix_kernel|seam_kernel|router|attn|rms_norm|"
                    r"elementwise|nvjet|splitKreduce|Kernel2|tilelang")
GROUPS = ["attention", "moe_hot", "gt_stage", "engram", "drafter", "head", "sampler", "other"]


def load_sqlite(path):
    q = ("select k.start, k.end, s.value, d.value from CUPTI_ACTIVITY_KIND_KERNEL k "
         "join StringIds s on k.shortName = s.id join StringIds d on k.demangledName = d.id order by k.start")
    return [(a, b, sn, dn) for a, b, sn, dn in sqlite3.connect(path).execute(q)]


def load_chrome(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    ev = d["traceEvents"] if isinstance(d, dict) else d
    out = []
    for e in ev:
        if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_kernel"):
            ts = float(e["ts"]) * 1e3  # us -> ns
            out.append((ts, ts + float(e.get("dur", 0)) * 1e3, e["name"].split("(")[0].split("<")[0].split("::")[-1], e["name"]))
    out.sort()
    return out


def key_of(sn, dn):
    return dn[:90] if sn in ("fwd_kernel", "kernel", "") else sn


def group_of(sn, dn):
    txt = sn + " | " + dn
    for g, rx in RULES:
        if rx.search(txt):
            return g
    return "other"


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    delim = opt("--delim", "_rejection_kernel")
    top = int(opt("--top", "60"))
    main_moe = opt("--main-moe", "mega_moe")
    min_steps = int(opt("--min-steps", "4"))
    K = load_sqlite(path) if path.endswith(".sqlite") else load_chrome(path)
    if not K:
        print("no kernel records in the trace (graph replay hidden? see the K5 note, section 1.4)")
        return 1
    cnt = collections.Counter(k[2] for k in K)
    bounds = [i for i, k in enumerate(K) if k[2] == delim or delim in k[3][:120] and k[2] in ("kernel", "fwd_kernel")]
    if len(bounds) < min_steps:
        moe = sum(v for n, v in cnt.items() if "mega_moe" in n)
        for name, v in sorted(cnt.items(), key=lambda x: -x[1]):
            if v > 3 and moe and moe % v == 0 and moe // v >= 2 and "mega_moe" not in name:
                bounds = [i for i, k in enumerate(K) if k[2] == name]
                print(f"# step delimiter inferred: {name} ({v} launches)")
                break
    if len(bounds) < min_steps:
        print(f"no per-step delimiter found (tried {delim}); pass --delim NAME. Most frequent names:")
        for n, v in cnt.most_common(25):
            print(f"  {v:8d}  {n[:100]}")
        return 2

    use_span = "--no-span" not in argv
    steps = []  # each: dict(wall, busy, sumdur, n, rows=[(key, group, dur)])
    for a, b in zip(bounds, bounds[1:]):
        seg = K[a:b]
        t0, t1 = seg[0][0], K[b][0]
        iv = sorted((s, min(e, t1)) for s, e, _, _ in seg)
        busy, cs, ce = 0, iv[0][0], iv[0][1]
        for s, e in iv[1:]:
            if s > ce:
                busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += ce - cs
        # main span by position
        lo, hi = 0, len(seg)
        gts = [i for i, k in enumerate(seg) if "gt_stage" in k[2]]
        if use_span and gts:
            moes = [i for i, k in enumerate(seg) if main_moe in k[2] or main_moe in k[3][:160]]
            after = [m for m in moes if m > gts[-1]]
            hi = (after[0] + 1) if after else len(seg)
            # layer prefix length: median count of kernels between a main MoE and the next gt_stage
            pre = []
            for g0, g1 in zip(gts, gts[1:]):
                mm = [m for m in moes if g0 < m < g1]
                if mm:
                    pre.append(g1 - mm[-1] - 1)
            lo = max(0, gts[0] - (int(st.median(pre)) if pre else 0))
        rows = []
        front = t0  # frontier attribution: a kernel is charged only for the time it extends the busy frontier.
        for i, (s, e, sn, dn) in enumerate(seg):
            g = group_of(sn, dn)
            if use_span and gts and not (lo <= i < hi) and g in ("attention", "moe_hot", "other") and SHARED.search(sn + dn):
                g = "drafter"
            e2 = min(e, t1)
            adv = max(0, e2 - max(front, s))
            front = max(front, e2)
            rows.append((key_of(sn, dn), g, e - s, adv))
        steps.append({"wall": t1 - t0, "busy": busy, "sumdur": sum(r[2] for r in rows), "n": len(seg), "rows": rows})
    steps = steps[1:-1] or steps
    n = len(steps)
    wall = st.median(s["wall"] for s in steps)
    print(f"trace={path}")
    print(f"steps={n}  step_wall_median_ms={wall/1e6:.3f}  mean_ms={st.mean(s['wall'] for s in steps)/1e6:.3f}  "
          f"gpu_busy_ms={st.median(s['busy'] for s in steps)/1e6:.3f} ({100*st.median(s['busy']/s['wall'] for s in steps):.1f}%)  "
          f"sum_of_kernel_ms={st.median(s['sumdur'] for s in steps)/1e6:.3f}  kernels/step={st.median(s['n'] for s in steps):.0f}")
    print("# gap test: busy close to wall means no idle; sum_of_kernel above busy means kernels overlap (side streams, PDL).")
    print("# tracing inflates every kernel a little: compare step_wall here with the untraced ms/step before trusting shares.")
    durs = [r[2] for s in steps for r in s["rows"]]
    print("\nduration histogram (per step)")
    for lo_, hi_ in ((0, 3e3), (3e3, 6e3), (6e3, 10e3), (10e3, 20e3), (20e3, 50e3), (50e3, 1e12)):
        sel = [d for d in durs if lo_ <= d < hi_]
        lab = f"{lo_/1e3:.0f}-{hi_/1e3:.0f} us" if hi_ < 1e11 else f">={lo_/1e3:.0f} us"
        print(f"  {lab:>10}: {len(sel)/n:7.1f} kernels  {sum(sel)/n/1e3:9.1f} us")
    u20 = [d for d in durs if d < 20e3]
    print(f"  under 20 us: {len(u20)/n:.1f} kernels/step, {sum(u20)/n/1e3:.1f} us/step")

    tot, calls, ftot = collections.Counter(), collections.Counter(), collections.Counter()
    gtot, gcalls, gft = collections.Counter(), collections.Counter(), collections.Counter()
    for s in steps:
        for key, g, d, adv in s["rows"]:
            k2 = (key, g)
            tot[k2] += d
            ftot[k2] += adv
            calls[k2] += 1
            gtot[g] += d
            gft[g] += adv
            gcalls[g] += 1
    wall_mean = sum(s["wall"] for s in steps) / n
    print("\n# us/step = sum of CUPTI durations (with PDL this includes time a kernel sat resident before its wait).")
    print("# frontier = time the kernel extended the busy frontier (what it adds to the serial chain). Rank fusions by frontier.")
    print("# share = frontier us/step over the mean step wall.")
    print(f"\n{'group':<10}{'calls/step':>11}{'us/step':>10}{'frontier':>10}{'share':>8}")
    for g in GROUPS:
        print(f"{g:<10}{gcalls[g]/n:>11.1f}{gtot[g]/n/1e3:>10.1f}{gft[g]/n/1e3:>10.1f}{100*gft[g]/n/wall_mean:>7.1f}%")
    print(f"{'(idle)':<10}{'':>11}{'':>10}{(wall_mean-sum(gft.values())/n)/1e3:>10.1f}{100*(1-sum(gft.values())/n/wall_mean):>7.1f}%")
    print(f"\n{'group':<10}{'calls/step':>11}{'us/step':>10}{'avg us':>8}{'frontier':>10}{'share':>8}  kernel")
    table = []
    for (key, g), v in sorted(ftot.items(), key=lambda x: -x[1]):
        c = calls[(key, g)]
        row = {"kernel": key, "group": g, "calls_per_step": round(c / n, 2),
               "us_per_step": round(tot[(key, g)] / n / 1e3, 2), "avg_us": round(tot[(key, g)] / c / 1e3, 2),
               "frontier_us_per_step": round(v / n / 1e3, 2), "share_pct": round(100 * v / n / wall_mean, 2)}
        table.append(row)
    for r in table[:top]:
        print(f"{r['group']:<10}{r['calls_per_step']:>11.1f}{r['us_per_step']:>10.1f}{r['avg_us']:>8.1f}"
              f"{r['frontier_us_per_step']:>10.1f}{r['share_pct']:>7.1f}%  {r['kernel'][:100]}")
    if len(table) > top:
        rest = table[top:]
        print(f"{'(rest)':<10}{sum(r['calls_per_step'] for r in rest):>11.1f}{sum(r['us_per_step'] for r in rest):>10.1f}"
              f"{'':>8}{sum(r['frontier_us_per_step'] for r in rest):>10.1f}{sum(r['share_pct'] for r in rest):>7.1f}%"
              f"  {len(rest)} more kernel names (use --top or --csv)")
    if "--csv" in argv:
        with open(opt("--csv"), "w") as f:
            f.write("group,calls_per_step,us_per_step,avg_us,frontier_us_per_step,share_pct,kernel\n")
            for r in table:
                f.write(f"{r['group']},{r['calls_per_step']},{r['us_per_step']},{r['avg_us']},{r['frontier_us_per_step']},"
                        f"{r['share_pct']},\"{r['kernel']}\"\n")
    if "--json" in argv:
        json.dump({"trace": path, "steps": n, "step_wall_median_ms": wall / 1e6,
                   "groups": {g: {"calls_per_step": gcalls[g] / n, "us_per_step": gtot[g] / n / 1e3,
                                    "frontier_us_per_step": gft[g] / n / 1e3} for g in GROUPS},
                   "kernels": table}, open(opt("--json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
