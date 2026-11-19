#!/bin/bash
# Stage 0b, part 1: build the Pg61 tree (Garner's engine + Meadlock's lane patches) on the Station.
# CPU only, no GPU, no RTX. Follows research/R1-pg61-build-runbook.md steps 0-5.
set -u
export JM=/raid/work/src/jm-recipes
export R=$JM/recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark-sidecar
export GARNER_SRC=/raid/work/src/garner
export WORK=/raid/work/pg61-work
export LOGS=/raid/work/logs
echo "== step 1: pre-checks"
for t in git patch docker sha1sum sha256sum find xargs; do command -v $t >/dev/null || echo "MISSING $t"; done
echo "garner HEAD: $(git -C $GARNER_SRC rev-parse HEAD)"
echo "garner dirty lines: $(git -C $GARNER_SRC status --porcelain | wc -l)   pycache dirs: $(find $GARNER_SRC/deepseek-v4.1-flash -name __pycache__ | wc -l)"
echo "jm HEAD: $(git -C $JM rev-parse --short HEAD)"
docker image inspect -f '{{.Id}} {{.Architecture}}' vllm/vllm-openai:v0.31.0 | cut -c1-40
if [ -e "$WORK/tree" ]; then echo "WORK tree already exists: $WORK (not rebuilding)"; else
  echo "== step 2: build"
  GARNER_SRC=$GARNER_SRC WORK=$WORK bash $R/scripts/launch-pg61.sh build 2>&1 | tee $LOGS/pg61-build-$(date +%Y%m%d-%H%M%S).log | cut -c1-200
fi
echo "== step 3: products"
ls $WORK | tr '\n' ' '; echo
echo "manifest lines: $(wc -l < $WORK/BUILD-SHA256SUMS) (want 116)"
echo "policy.env sha256: $(sha256sum $WORK/policy.env | cut -c1-16) (want 1dad6e12c09064c2)"
echo "overlay files: $(find $WORK/overlay-vllm -type f | wc -l) (want 9)"
( cd $WORK/tree && sha1sum serve/serve.sh config/recipe.env fastpath/w5_markov.py fastpath/w5_markov_gemv.py kernels/w6-moe/markov/w6mk.cu | cut -c1-8,41- | tr '\n' ' ' ); echo
echo "   want: 212d2a03 serve.sh, 71d39119 recipe.env, 62357c0b w5_markov.py, ede5fd52 w5_markov_gemv.py, a17b2ba2 w6mk.cu"
echo "== step 4: manifest check (expect only policy.env and libw6mk.so to FAIL)"
M=$(realpath $R/results/2026-10-29-pg61-garner-engine/BUILD-SHA256SUMS.expected)
( cd $WORK && sha256sum -c --quiet "$M" 2>&1 | cut -c1-160 ); echo "rc=$?"
echo "unexpected lines (should be none):"
( cd $WORK && sha256sum -c --quiet "$M" 2>&1 | grep -v -E '^(policy\.env|tree/fastpath/libw6mk\.so): FAILED$|WARNING' )
echo "== step 5: device-code identity of libw6mk.so"
bash $R/scripts/w1_cubin_hash.sh $WORK/tree/fastpath/libw6mk.so 2>&1 | tail -4 | cut -c1-160
echo "   want: .1 11b55da203679618..., .2 4d07c09332118801..."
echo "== policy.env"; cat $WORK/policy.env | cut -c1-200
