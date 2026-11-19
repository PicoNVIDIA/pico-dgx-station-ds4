#!/bin/bash
# Apply K7 patches 0002 (host) and 0003 (kernel edit list) to the engine tree, idempotent, and add the serve.sh line.
set -u
GT=/raid/work/gt/gracetier; K7=/raid/work/run/patches/K7
cd "$GT" || exit 1
FILES="tree/fastpath/fastpath_a12_memory_guard.py tree/fastpath/fastpath_a12_retire.py tree/fastpath/fastpath_gracetier.py tree/fastpath/fastpath_megamoe.py tree/fastpath/gracetier_rt/runtime/gracetier_res.py tree/fastpath/gracetier_rt/runtime/gracetier_stage.py tree/serve/serve.sh"
[ -f "$K7/pre.sha" ] || { [ -f tree/fastpath/megamoe_sfpack.py ] || sha256sum $FILES > "$K7/pre.sha"; }
[ -f "$K7/jit-root-before.txt" ] || { L=$(ls -t "$GT"/home/state/logs/*-ds41.log 2>/dev/null | head -1); [ -n "$L" ] && tr -d '\r' < "$L" | grep -m1 "JIT include root" | sed -E 's/^.*(megamoe[^:]*: .*JIT include root.*)$/\1/' | cut -c1-300 > "$K7/jit-root-before.txt"; }
for p in 0002-gracetier-sf-packed-host.patch 0003-megamoe-sf-packed-kernel.patch; do
  if patch -p1 -N --dry-run < "$K7/$p" > /dev/null 2>&1; then patch -p1 -N < "$K7/$p" > /dev/null && echo "$p applied"
  elif patch -p1 -R --dry-run < "$K7/$p" > /dev/null 2>&1; then echo "$p already applied"
  else echo "K7 PATCH $p FAILED TO APPLY"; patch -p1 -N --dry-run < "$K7/$p" 2>&1 | grep -E "FAILED|rejects|can't find" | head -n 4; exit 1; fi
done
bash "$K7/k7_serve_insert.sh" tree/serve/serve.sh || exit 1
bash -n tree/serve/serve.sh || exit 1
for f in $FILES tree/fastpath/megamoe_sfpack.py tree/fastpath/gracetier_rt/runtime/gracetier_sfpack.py; do
  case "$f" in *.py) python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" || { echo "K7: $f does not parse"; exit 1; } ;; esac
done
echo "K7 in place, syntax ok"
