#!/usr/bin/env python3
"""Compare two identity captures. Reports identical prompts, where each differing prompt first diverges,
and how close the top two tokens were at that position in run A (a small margin = a rounding-level flip).
usage: ident_cmp.py <a.json> <b.json>"""
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ra = {(r["set"], r["i"]): r for r in a["rows"]}; rb = {(r["set"], r["i"]): r for r in b["rows"]}
keys = [k for k in ra if k in rb and "error" not in ra[k] and "error" not in rb[k]]
same = 0; tok_same = 0; tok_total = 0; div = []
for k in keys:
    ta, tb = ra[k]["tokens"], rb[k]["tokens"]
    use_tokens = bool(ta) and bool(tb)
    xa, xb = (ta, tb) if use_tokens else (ra[k]["text"], rb[k]["text"])
    n = min(len(xa), len(xb)); j = 0
    while j < n and xa[j] == xb[j]:
        j += 1
    tok_total += max(len(xa), len(xb)); tok_same += j
    if j == len(xa) == len(xb):
        same += 1
    else:
        m = None
        if use_tokens and j < len(ra[k]["lp1"]) and ra[k]["lp2"][j] is not None:
            m = round(ra[k]["lp1"][j] - ra[k]["lp2"][j], 4)
        div.append((k, j, len(xa), len(xb), m, (xa[j] if j < len(xa) else None), (xb[j] if j < len(xb) else None)))
unit = "tokens" if any(ra[k]["tokens"] for k in keys) else "chars"
print(f"A={a['model']}  B={b['model']}  prompts compared {len(keys)}, errors A {len(a['rows'])-sum('error' not in r for r in a['rows'])} B {len(b['rows'])-sum('error' not in r for r in b['rows'])}")
print(f"identical outputs: {same}/{len(keys)}; matching prefix {tok_same}/{tok_total} {unit} ({100*tok_same/max(tok_total,1):.1f} %)")
for k, j, la, lb, m, xa, xb in div[:40]:
    print(f"  {k[0]}{k[1]:02d}: first difference at {unit[:-1]} {j} (lengths {la}/{lb}); top1-top2 margin in A {m}; A={xa!r} B={xb!r}")
