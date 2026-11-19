#!/usr/bin/env python3
"""Build hot maps from a residency dump (gracetier_res.dump: dump-<tag>.json + .npz) and say what they would cover.

usage: fitmap.py <dump.json> <dump.npz> <N hot per layer> <boot hot map json> <out prefix>
writes <out prefix>cnt<N>.json   top-N experts per layer by the cumulative pick counter of the dumped boot
                                 (ties: experts the manager holds hot now first, then expert id)
       <out prefix>res<N>.json   the hot set the residency manager held when the dump was taken
Both are {"<layer>": [sorted expert ids]}, the format of docs/maps/hot-*.json.
The counter is int [L, E], row l = layer l, column = expert id, one count per pick of a decode call, hot or cold
(runtime/gracetier_stage.py; calls with very many ids, such as prefill chunks, are not counted).
Coverage = the share of the dumped boot's picks that fall on a map's hot experts; 1 - coverage is the cold share that
map would have had on the same traffic if nothing moved.
"""
import json
import sys

import numpy as np

dump_json, dump_npz, n, boot_map, prefix = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
d = json.load(open(dump_json))
cnt = np.load(dump_npz)["counts"].astype(np.int64)
boot = {int(k): [int(e) for e in v] for k, v in json.load(open(boot_map)).items()}
layers = sorted(int(l) for l in d["layers"])
E = int(d.get("E", cnt.shape[1]))
assert cnt.shape[1] == E and cnt.shape[0] > max(layers), (cnt.shape, E, max(layers))
by_cnt, by_res, cov = {}, {}, {"boot": [0, 0], "res": [0, 0], "cnt": [0, 0]}
ov_boot, ov_res, zero = [], [], []
for l in layers:
    cur = [int(e) for e in d["layers"][str(l)]["hot"]]
    assert len(cur) == n and len(set(cur)) == n, (l, len(cur), n)
    assert len(boot[l]) == n, (l, len(boot[l]))
    c, held = cnt[l], set(cur)
    order = sorted(range(E), key=lambda e: (-int(c[e]), e not in held, e))
    top = sorted(order[:n])
    by_cnt[str(l)], by_res[str(l)] = top, sorted(cur)
    for name, s in (("boot", boot[l]), ("res", cur), ("cnt", top)):
        cov[name][0] += int(c[list(s)].sum()); cov[name][1] += int(c.sum())
    ov_boot.append(len(set(top) & set(boot[l]))); ov_res.append(len(set(top) & held))
    zero.append(int((c[top] == 0).sum()))
json.dump(by_cnt, open(f"{prefix}cnt{n}.json", "w"))
json.dump(by_res, open(f"{prefix}res{n}.json", "w"))
print(f"dump {d.get('tag')}: {len(layers)} layers, {n} hot of {E}, host steps {d.get('host_steps')}, swaps {d.get('swaps')}, "
      f"hot_moved {d.get('hot_moved')}, picks counted {int(cnt[layers].sum()):,}")
for name, label in (("boot", "boot map"), ("res", "manager's hot set at the dump"), ("cnt", f"top-{n} by count")):
    a, b = cov[name]
    print(f"  {label:32s} covers {100.0 * a / max(1, b):6.2f} % of the picks (cold share {100.0 * (1 - a / max(1, b)):5.2f} %)")
print(f"  top-{n} by count: shares {np.mean(ov_boot):.1f} experts a layer with the boot map (min {min(ov_boot)}), "
      f"{np.mean(ov_res):.1f} with the manager's set (min {min(ov_res)}); never-picked experts in it {np.mean(zero):.1f} a layer")
print(f"  wrote {prefix}cnt{n}.json and {prefix}res{n}.json")
