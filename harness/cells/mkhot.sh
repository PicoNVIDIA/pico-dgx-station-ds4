#!/bin/bash
# Pre-hook helper: write a hot map with N hot experts per layer and rebuild the engine's row map from it.
# usage: mkhot.sh <N>   (N >= 275). The extra experts are the lowest-numbered cold ones of the shipped 275 map;
# the residency manager swaps in the experts the traffic uses during the warm-up, so the choice only seeds it.
set -eu
N=${1:?hot experts per layer}
GT=/raid/work/gt/gracetier; DEC=/raid/work/src/gb300-dsv41-decode
OUT=/raid/work/run/cells/hot-$N.json
python3 - "$DEC/docs/maps/hot-garner255-plus-275.json" "$N" "$OUT" <<'EOF'
import json, sys
src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(src))
res = {}
for layer, hot in m.items():
    hot = list(hot)
    assert len(hot) == 275 and len(set(hot)) == 275, (layer, len(hot))
    have = set(hot)
    extra = [e for e in range(384) if e not in have][: n - 275]
    assert len(extra) == n - 275 >= 0
    res[layer] = hot + extra
json.dump(res, open(out, "w"))
print(f"hot map: {len(res)} layers x {n} -> {out}")
EOF
python3 "$DEC/port/garner/make_rowmap_allres.py" --rowmap "$GT/tree/fastpath/hook/rowmap-mix-v1-h255.json" \
  --out "$GT/tree/fastpath/hook/rowmap-allres.json" --hot-map "$OUT" 2>&1 | cut -c1-160
