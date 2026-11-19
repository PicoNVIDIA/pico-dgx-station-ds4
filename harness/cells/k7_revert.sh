#!/bin/bash
# Take the K7 patches out again and check the tree is back to what k7_apply.sh first saw.
set -u
GT=/raid/work/gt/gracetier; K7=/raid/work/run/patches/K7
cd "$GT" || exit 1
bash "$K7/k7_serve_insert.sh" --remove tree/serve/serve.sh
for p in 0003-megamoe-sf-packed-kernel.patch 0002-gracetier-sf-packed-host.patch; do
  if patch -p1 -R --dry-run < "$K7/$p" > /dev/null 2>&1; then patch -p1 -R < "$K7/$p" > /dev/null && echo "$p reversed"
  else echo "$p was not applied (or does not reverse cleanly)"; fi
done
for f in tree/fastpath/megamoe_sfpack.py tree/fastpath/gracetier_rt/runtime/gracetier_sfpack.py; do [ -s "$f" ] && echo "LEFT BEHIND: $f"; [ -f "$f" ] && [ ! -s "$f" ] && rm -f "$f"; done
if [ -f "$K7/pre.sha" ]; then sha256sum -c "$K7/pre.sha" > /tmp/k7-revert.chk 2>&1 && echo "tree restored: every touched file matches its pre-K7 hash" || { echo "TREE NOT RESTORED:"; grep -v ": OK" /tmp/k7-revert.chk | head -n 8; }; fi
