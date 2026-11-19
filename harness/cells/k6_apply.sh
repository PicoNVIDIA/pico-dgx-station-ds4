#!/bin/bash
# Apply the K6 patch to the engine tree (idempotent) and forward its three variables in serve.sh.
set -u
GT=/raid/work/gt/gracetier; P=/raid/work/run/patches/K6/0001-gracetier-stage-skip.patch
cd "$GT" || exit 1
if patch -p1 -N --dry-run < "$P" > /dev/null 2>&1; then patch -p1 -N < "$P" | tr '\n' ' '; echo "-> K6 applied"
elif patch -p1 -R --dry-run < "$P" > /dev/null 2>&1; then echo "K6 already applied"
elif [ -f tree/fastpath/megamoe_sfpack.py ] && grep -q "STAGE_SKIP" tree/fastpath/fastpath_gracetier.py; then echo "K6 already applied (under the K7 patches, which change the same files, so it cannot be reverse-checked)"
else echo "K6 PATCH FAILED TO APPLY"; exit 1; fi
python3 - <<'PY'
p = "/raid/work/gt/gracetier/tree/serve/serve.sh"
s = open(p).read()
anchor = '  [ -n "${DS41_PEER3_SPINS:-}" ] && envs+=(-e "PEER3_SPINS=$DS41_PEER3_SPINS")\n'
lines = ('  [ -n "${DS41_GT_STAGE_SKIP:-}" ] && envs+=(-e "GRACETIER_STAGE_SKIP=$DS41_GT_STAGE_SKIP")\n'
         '  [ -n "${DS41_GT_SCRATCH_LANES:-}" ] && envs+=(-e "GRACETIER_SCRATCH_LANES=$DS41_GT_SCRATCH_LANES")\n'
         '  [ -n "${DS41_GT_STAGE_SKIP_AUDIT:-}" ] && envs+=(-e "GRACETIER_STAGE_SKIP_AUDIT=$DS41_GT_STAGE_SKIP_AUDIT")\n')
if "GRACETIER_STAGE_SKIP=" not in s:
    assert s.count(anchor) == 1, "anchor line not found exactly once"
    open(p, "w").write(s.replace(anchor, anchor + lines))
    print("serve.sh: K6 env lines added")
else:
    print("serve.sh: K6 env lines already present")
PY
bash -n tree/serve/serve.sh || exit 1
for f in tree/fastpath/fastpath_gracetier.py tree/fastpath/gracetier_rt/runtime/gracetier_stage.py tree/fastpath/gracetier_rt/runtime/gracetier_res.py; do
  python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" || exit 1
done
echo "K6 in place, syntax ok"
