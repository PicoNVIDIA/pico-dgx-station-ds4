#!/usr/bin/env python3
"""Diagnose a divergence between two identity captures (A = reference, B = candidate).
Answers: are the prompts tokenised the same; is the visible text the same even when token strings differ;
how large is the top1-top2 margin where they first split; is each engine's pick the other's runner-up.
usage: ident_diag.py <a.json> <b.json>"""
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ra = {(r["set"], r["i"]): r for r in a["rows"]}; rb = {(r["set"], r["i"]): r for r in b["rows"]}
keys = [k for k in ra if k in rb and "error" not in ra[k] and "error" not in rb[k]]
pt_same = sum((ra[k]["usage"] or {}).get("prompt_tokens") == (rb[k]["usage"] or {}).get("prompt_tokens") for k in keys)
txt_same = [k for k in keys if ra[k]["text"] == rb[k]["text"]]
print(f"prompts {len(keys)}; same prompt_tokens {pt_same}/{len(keys)}; same visible text {len(txt_same)}/{len(keys)} {[f'{k[0]}{k[1]:02d}' for k in txt_same]}")
ma, mb, cross, pos = [], [], 0, []
n_div = 0
for k in keys:
    if ra[k]["text"] == rb[k]["text"]:
        continue
    ta, tb = ra[k]["tokens"], rb[k]["tokens"]
    j = 0
    while j < min(len(ta), len(tb)) and ta[j] == tb[j]:
        j += 1
    if j >= min(len(ta), len(tb)):
        continue
    n_div += 1; pos.append(j)
    if ra[k]["lp2"][j] is not None: ma.append(ra[k]["lp1"][j] - ra[k]["lp2"][j])
    if rb[k]["lp2"][j] is not None: mb.append(rb[k]["lp1"][j] - rb[k]["lp2"][j])
    t2 = (rb[k].get("tok2") or [None] * (j + 1))[j]
    if t2 is not None and t2 == ta[j]: cross += 1
def dist(m):
    m = sorted(m)
    if not m: return "n/a"
    b_ = [0, 0.126, 0.51, 1.01, 2.01, 99]
    cnt = [sum(lo <= x < hi for x in m) for lo, hi in zip(b_, b_[1:])]
    return f"n={len(m)} median {m[len(m)//2]:.3f} max {m[-1]:.3f}; <=0.125: {cnt[0]}, 0.25-0.5: {cnt[1]}, 0.625-1.0: {cnt[2]}, 1.1-2.0: {cnt[3]}, >2.0: {cnt[4]}"
pos.sort()
print(f"real divergences {n_div}; first-difference position median {pos[len(pos)//2] if pos else None}, min {pos[0] if pos else None}, max {pos[-1] if pos else None}")
print("margin (top1-top2 logprob) at the split, in A:", dist(ma))
print("margin (top1-top2 logprob) at the split, in B:", dist(mb))
print(f"B's runner-up is A's pick at the split: {cross}/{n_div}")
steps = sorted({round(abs(x), 3) for r in list(ra.values())[:5] for x in r["lp1"][:60]})[:12]
print("smallest distinct |logprob| values in A (shows the logprob grid):", steps)
