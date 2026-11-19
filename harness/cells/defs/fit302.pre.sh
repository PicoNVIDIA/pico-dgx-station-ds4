#!/bin/bash
bash /raid/work/run/cells/k6_apply.sh || exit 1
GT=/raid/work/gt/gracetier; DEC=/raid/work/src/gb300-dsv41-decode; M=/raid/work/run/cells/hot-fitcnt302.json
[ -s "$M" ] || { echo "NO FITTED MAP $M: this boot runs the base map, discard it"; exit 1; }
python3 "$DEC/port/garner/make_rowmap_allres.py" --rowmap "$GT/tree/fastpath/hook/rowmap-mix-v1-h255.json" \
  --out "$GT/tree/fastpath/hook/rowmap-allres.json" --hot-map "$M" 2>&1 | cut -c1-160
