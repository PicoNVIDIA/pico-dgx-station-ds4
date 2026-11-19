#!/usr/bin/env python3
"""K4: offline analysis of the gracetier stage log (written by 0001-gracetier-stage-log.patch).

Input: the JSONL file named by GRACETIER_STAGE_LOG. One header line ({"event": "open", ...}), then one line per
engine step:  {"s": step, "t": time, "ntok": n, "nreq": n, "swaps": n,
               "cold":  {layer: [expert, slot, picks, ...]},      # cold experts gt_stage copied in that step
               "picks": {layer: [expert, n, ...]}}                # optional: every routed pick (hot and cold)

Reports (overall, and per bucket of concurrent requests `nreq` when the log has it):
  1. census      distinct cold experts staged per layer-call and per step; modeled staging ms per step
  2. repeat      share of a step's staged experts that the same layer also staged in the previous 1/2/4/8/16 steps
  3. LRU         hit rate of a per-layer LRU staged cache of 4/8/16/32 experts (design c, private memory)
  4. scratch     hit rate of "skip the copy if the shared scratch slot still holds this layer's expert"
                 (design c1, zero extra HBM), with the scratch split into 1/2/4/8 lanes (layer mod lanes)
  5. predictors  recall and waste of: previous-W-steps picks (W = 1, 2, 4, 8) and top-N by a decayed counter
                 (N = 1, 2, 4, 8, 16), per layer (design b)
  6. extra hot   share of staged copies that k more hot experts per layer would remove (static, fitted on the first
                 half of the log and scored on the second half), uniform and water-filled over layers: the control
                 for 3 at the same HBM cost

Nothing here needs torch or numpy. Usage:
  python3 k4_stage_analyze.py LOG [--skip N] [--max-ntok N] [--nreq-min N] [--nreq-max N] [--us-per-expert 48.7]
                                  [--fixed-us-per-layer 6] [--halflife 64] [--json OUT] [--per-layer]
"""
from __future__ import annotations

import argparse
import collections
import heapq
import json
import sys

WINDOWS = (1, 2, 4, 8, 16)
LRU_K = (4, 8, 16, 32)
LANES = (1, 2, 4, 8)
PRED_W = (1, 2, 4, 8)
PRED_N = (1, 2, 4, 8, 16)
EXTRA_K = (4, 8, 16, 32)
GROUP_MIB = (2 * 2304 * (5120 // 2) + 5120 * (2304 // 2)) / 2 ** 20   # L1 + L2 bytes of one expert group (weights)


def load(path):
    """-> (header, steps) of the LAST segment of the file (a restart appends a new header)."""
    hdr, steps, segs = None, [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue                      # a torn last line
            if r.get("event") == "open":
                hdr, steps = r, []
                segs += 1
                continue
            steps.append(r)
    if hdr is None:
        raise SystemExit(f"{path}: no header line (not a K4 stage log?)")
    return hdr, steps, segs


def triples(flat):
    return [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat) - 2, 3)]


def bucket(nreq):
    if nreq is None:
        return "nreq=?"
    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32)):
        if lo <= nreq <= hi:
            return f"nreq={lo}" if lo == hi else f"nreq={lo}-{hi}"
    return "nreq>32"


def pct(a, b):
    return "   n/a" if not b else f"{100.0 * a / b:6.1f}"


class Acc:
    """Counters of one bucket."""

    def __init__(self):
        self.steps = 0
        self.calls = 0                 # layer-calls
        self.zero_calls = 0
        self.staged = 0                # distinct cold experts staged, summed over layer-calls
        self.picks_cold = 0
        self.max_call = 0
        self.rep = collections.Counter()        # W -> staged experts also staged in the previous W steps
        self.lru = collections.Counter()        # K -> hits
        self.lane = collections.Counter()       # lanes -> hits
        self.pw = {w: [0, 0, 0] for w in PRED_W}    # W -> [hit, predicted, actual]
        self.pn = {n: [0, 0, 0] for n in PRED_N}


def analyze(hdr, steps, a):
    layers = [int(l) for l in hdr["layers"]]
    Hs = {int(k): v for k, v in hdr["H"].items()}
    accs = collections.defaultdict(Acc)
    last_seen = {l: {} for l in layers}                 # layer -> {expert: step position}
    lru = {k: {l: collections.OrderedDict() for l in layers} for k in LRU_K}
    owner = {n: {} for n in LANES}                      # lanes -> {(lane, slot): (layer, expert)}
    ewma = {l: collections.defaultdict(float) for l in layers}
    decay = 0.5 ** (1.0 / max(1e-9, a.halflife))
    per_layer = {l: {"calls": 0, "staged": 0, "rep1": 0, "rep8": 0, "lru8": 0, "lru32": 0} for l in layers}
    freq_half = [{l: collections.Counter() for l in layers} for _ in (0, 1)]   # steps staged, per half
    half = len(steps) // 2
    for pos, r in enumerate(steps):
        cold = {int(l): triples(v) for l, v in (r.get("cold") or {}).items()}
        names = ("all", bucket(r.get("nreq")))
        for nm in names:
            accs[nm].steps += 1
        for l in layers:
            tr = cold.get(l, ())
            A = {e: n for e, _c, n in tr}
            ls = last_seen[l]
            # ---- predictors, evaluated before this step's picks are known
            pw_sets = {w: 0 for w in PRED_W}
            pw_hit = {w: 0 for w in PRED_W}
            rep = {w: 0 for w in WINDOWS}
            for e, p0 in ls.items():
                age = pos - p0
                for w in PRED_W:
                    if age <= w:
                        pw_sets[w] += 1
                        if e in A:
                            pw_hit[w] += 1
            for e in A:
                p0 = ls.get(e)
                if p0 is not None:
                    for w in WINDOWS:
                        if pos - p0 <= w:
                            rep[w] += 1
            ew = ewma[l]
            top = heapq.nlargest(max(PRED_N), ew.items(), key=lambda kv: kv[1]) if ew else []
            pn_pred = {n: min(n, len(top)) for n in PRED_N}
            pn_hit = {n: sum(1 for e, _ in top[:n] if e in A) for n in PRED_N}
            lru_hit = {k: sum(1 for e in A if e in lru[k][l]) for k in LRU_K}
            lane_hit = {}
            for n in LANES:
                ow, h = owner[n], 0
                for e, c, _n in tr:
                    key = (l % n, c)
                    if ow.get(key) == (l, e):
                        h += 1
                    ow[key] = (l, e)
                lane_hit[n] = h
            # ---- accumulate
            for nm in names:
                ac = accs[nm]
                ac.calls += 1
                ac.zero_calls += not A
                ac.staged += len(A)
                ac.picks_cold += sum(A.values())
                ac.max_call = max(ac.max_call, len(A))
                for w in WINDOWS:
                    ac.rep[w] += rep[w]
                for k in LRU_K:
                    ac.lru[k] += lru_hit[k]
                for n in LANES:
                    ac.lane[n] += lane_hit[n]
                for w in PRED_W:
                    ac.pw[w][0] += pw_hit[w]
                    ac.pw[w][1] += pw_sets[w]
                    ac.pw[w][2] += len(A)
                for n in PRED_N:
                    ac.pn[n][0] += pn_hit[n]
                    ac.pn[n][1] += pn_pred[n]
                    ac.pn[n][2] += len(A)
            pl = per_layer[l]
            pl["calls"] += 1
            pl["staged"] += len(A)
            pl["rep1"] += rep[1]
            pl["rep8"] += rep[8]
            pl["lru8"] += lru_hit[8]
            pl["lru32"] += lru_hit[32]
            # ---- update the state with this step's picks
            for e in A:
                ls[e] = pos
                freq_half[pos >= half][l][e] += 1
            for k in LRU_K:
                od = lru[k][l]
                for e, _n in sorted(A.items(), key=lambda kv: kv[1]):     # most-picked last = most recent
                    od[e] = pos
                    od.move_to_end(e)
                while len(od) > k:
                    od.popitem(last=False)
            if ew:
                for e in list(ew):
                    ew[e] *= decay
                    if ew[e] < 1e-4:
                        del ew[e]
            for e, n in A.items():
                ew[e] += n
    # ---- extra hot: fit on the first half, score on the second
    extra = {}
    tot2 = sum(sum(c.values()) for c in freq_half[1].values())
    for k in EXTRA_K:
        uni = sum(sum(freq_half[1][l][e] for e, _ in freq_half[0][l].most_common(k)) for l in layers)
        cand = sorted(((n, l, e) for l in layers for e, n in freq_half[0][l].items()), reverse=True)[:k * len(layers)]
        wf = sum(freq_half[1][l][e] for _n, l, e in cand)
        wf_layers = collections.Counter(l for _n, l, _e in cand)
        extra[k] = {"uniform_removed": uni, "waterfill_removed": wf, "scored_copies": tot2,
                    "hbm_GiB": round(k * len(layers) * GROUP_MIB / 1024, 2),
                    "waterfill_max_per_layer": max(wf_layers.values()) if wf_layers else 0}
    return accs, per_layer, extra, Hs


def report(hdr, steps, segs, accs, per_layer, extra, Hs, a, out=sys.stdout):
    w = lambda s="": print(s, file=out)
    nl = len(hdr["layers"])
    w(f"K4 stage log: {len(steps)} steps analysed, {nl} layers, H {sorted(set(Hs.values()))}, residency "
      f"{hdr.get('residency')}, picks {hdr.get('picks')}, segments in file {segs} (last one used)")
    if len(set(Hs.values())) > 1:
        w("  NOTE: H differs between layers: the 'scratch' rows assume slot c of every layer aliases one physical "
          "slot, which holds only for equal H.")
    sw = [r.get("swaps", 0) for r in steps]
    if sw:
        w(f"  residency swaps during the log: {sw[-1] - sw[0]}")
    order = ["all"] + sorted(k for k in accs if k != "all")
    for nm in order:
        ac = accs[nm]
        if ac.steps < a.min_steps and nm != "all":
            continue
        per_step = ac.staged / max(1, ac.steps)
        ms = (per_step * a.us_per_expert + nl * a.fixed_us_per_layer) / 1000.0
        w()
        w(f"== {nm}: {ac.steps} steps ==")
        w(f"1. census: {ac.staged / max(1, ac.calls):.2f} cold experts staged per layer-call (max {ac.max_call}, "
          f"{pct(ac.zero_calls, ac.calls).strip()} % of layer-calls stage none); {per_step:.1f} per step; cold picks "
          f"per staged expert {ac.picks_cold / max(1, ac.staged):.2f}")
        w(f"   modeled serial staging: {ms:.2f} ms/step ({a.us_per_expert} us/expert + {a.fixed_us_per_layer} "
          f"us/layer fixed) [model, not a measurement]")
        w("2. repeat rate, % of staged experts the same layer also staged in the previous W steps:")
        w("   " + "  ".join(f"W={x}: {pct(ac.rep[x], ac.staged).strip()}" for x in WINDOWS))
        w("3. per-layer LRU staged cache (private memory), hit % of staged experts [HBM GiB for all layers]:")
        w("   " + "  ".join(f"K={k}: {pct(ac.lru[k], ac.staged).strip()} [{k * nl * GROUP_MIB / 1024:.1f}]"
                             for k in LRU_K))
        w("4. shared-scratch persistence (zero extra HBM at 1 lane), hit % of staged experts:")
        w("   " + "  ".join(f"lanes={n}: {pct(ac.lane[n], ac.staged).strip()}" for n in LANES))
        w("5. predictors (recall = % of staged experts predicted; waste = % of predicted not staged; "
          "size = predicted per layer-call):")
        for x in PRED_W:
            h, p, t = ac.pw[x]
            w(f"   prev-{x:<2d} steps      recall {pct(h, t)}  waste {pct(p - h, p)}  size {p / max(1, ac.calls):6.2f}")
        for n in PRED_N:
            h, p, t = ac.pn[n]
            w(f"   counter top-{n:<2d}     recall {pct(h, t)}  waste {pct(p - h, p)}  size {p / max(1, ac.calls):6.2f}")
        if nm == "all":
            w(f"   (counter = decayed pick count of staged experts, half-life {a.halflife} steps)")
    w()
    w("6. extra hot experts per layer (static; fitted on the first half of the log, scored on the second half):")
    for k, v in extra.items():
        w(f"   +{k:<2d} per layer [{v['hbm_GiB']} GiB]: uniform removes {pct(v['uniform_removed'], v['scored_copies']).strip()} % "
          f"of staged copies; water-filled over layers {pct(v['waterfill_removed'], v['scored_copies']).strip()} % "
          f"(max {v['waterfill_max_per_layer']} in one layer)")
    w("   Compare row k here with LRU K = k in section 3: same HBM. The LRU is only worth building if it wins.")
    if a.per_layer:
        w()
        w("per layer: staged/call, repeat W=1 %, W=8 %, LRU8 hit %, LRU32 hit %")
        for l in sorted(per_layer):
            p = per_layer[l]
            w(f"   L{l:<3d} {p['staged'] / max(1, p['calls']):6.2f}  {pct(p['rep1'], p['staged'])}  "
              f"{pct(p['rep8'], p['staged'])}  {pct(p['lru8'], p['staged'])}  {pct(p['lru32'], p['staged'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N steps (warm-up)")
    ap.add_argument("--max-ntok", type=int, default=0, help="drop steps with more scheduled tokens (prefill chunks)")
    ap.add_argument("--nreq-min", type=int, default=0)
    ap.add_argument("--nreq-max", type=int, default=0)
    ap.add_argument("--us-per-expert", type=float, default=48.7)
    ap.add_argument("--fixed-us-per-layer", type=float, default=6.0)
    ap.add_argument("--halflife", type=float, default=64.0, help="decayed-counter half-life, steps")
    ap.add_argument("--min-steps", type=int, default=50, help="smallest bucket that is printed")
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    hdr, steps, segs = load(a.log)
    n0 = len(steps)
    steps = steps[a.skip:]
    if a.max_ntok:
        steps = [r for r in steps if not isinstance(r.get("ntok"), int) or r["ntok"] <= a.max_ntok]
    if a.nreq_min:
        steps = [r for r in steps if isinstance(r.get("nreq"), int) and r["nreq"] >= a.nreq_min]
    if a.nreq_max:
        steps = [r for r in steps if isinstance(r.get("nreq"), int) and r["nreq"] <= a.nreq_max]
    if not steps:
        raise SystemExit(f"{a.log}: no steps left after the filters ({n0} in the file)")
    ntok = collections.Counter(r.get("ntok") for r in steps)
    accs, per_layer, extra, Hs = analyze(hdr, steps, a)
    report(hdr, steps, segs, accs, per_layer, extra, Hs, a)
    print(f"\nscheduled tokens per step (value: steps), top 8: {ntok.most_common(8)}")
    print("Removed steps (filters) are treated as if they never ran: use --max-ntok to keep prefill out of the "
          "repeat and cache numbers, or leave it in to see what a real mixed run does.")
    if a.json:
        doc = {"steps": len(steps), "extra_hot": extra,
               "buckets": {nm: {"steps": ac.steps, "calls": ac.calls, "staged": ac.staged,
                                "repeat": dict(ac.rep), "lru": dict(ac.lru), "scratch_lanes": dict(ac.lane),
                                "pred_prev": ac.pw, "pred_counter": ac.pn} for nm, ac in accs.items()},
               "per_layer": per_layer}
        with open(a.json, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
