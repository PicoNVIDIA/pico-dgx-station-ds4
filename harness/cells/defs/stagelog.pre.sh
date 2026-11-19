#!/bin/bash
set -u
GT=/raid/work/gt/gracetier; P=/raid/work/run/patches/K4
bash /raid/work/run/cells/mkhot.sh 285
cd "$GT" || exit 1
if patch -p1 --dry-run < $P/0001-gracetier-stage-log.patch > /dev/null 2>&1; then patch -p1 < $P/0001-gracetier-stage-log.patch | tail -1
else echo "K4-0001 does not apply cleanly (already applied?)"; fi
python3 - <<'PY'
p = "/raid/work/gt/gracetier/tree/serve/serve.sh"
s = open(p).read()
anchor = '  [ -n "${DS41_PEER3_SPINS:-}" ] && envs+=(-e "PEER3_SPINS=$DS41_PEER3_SPINS")\n'
line = '  [ -n "${DS41_GT_STAGE_LOG:-}" ] && envs+=(-e "GRACETIER_STAGE_LOG=$DS41_GT_STAGE_LOG" -e "GRACETIER_STAGE_LOG_PICKS=${DS41_GT_STAGE_LOG_PICKS:-0}")\n'
if "GRACETIER_STAGE_LOG=" not in s:
    assert s.count(anchor) == 1, "anchor line not found exactly once"
    open(p, "w").write(s.replace(anchor, anchor + line))
    print("serve.sh: stage-log env line added")
PY
rm -f "$GT/home/state/prof/gracetier-res/k4-stage.jsonl"
bash -n tree/serve/serve.sh && python3 -c "import ast; ast.parse(open('tree/fastpath/fastpath_gracetier.py').read())" && echo "stage-log patch in place, syntax ok"
