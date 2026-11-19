#!/bin/bash
# Stage 0b, part 2 (prep only, no GPU work): layout and patched card for the gracetier boot.
# Follows research/R2-gracetier-launch-runbook.md steps 0, 1, 2, 4, 5. Does NOT stop the baseline or boot.
set -u
export GTROOT=/raid/work/gt
export GR=$GTROOT/garner-shim
export GT=$GTROOT/gracetier
export DEC=/raid/work/src/gb300-dsv41-decode
export MODEL_DIR=/raid/models/DeepSeek-V4.1-Flash-1017
export JM=/raid/work/src/jm-recipes/recipes/dgx-station-gb300
export PG61_WORK_REAL=/raid/work/pg61-work
export GT1_SCRATCH=/raid/work/logs/gt1
mkdir -p $GTROOT/card $GT1_SCRATCH
echo "== step 1: preconditions"
for t in flock docker nvidia-smi python3 rsync patch git curl timeout openssl numactl; do command -v $t >/dev/null || echo "MISSING $t"; done
ls $PG61_WORK_REAL/tree/serve/serve.sh $PG61_WORK_REAL/tree/fastpath/libw6mk.so $PG61_WORK_REAL/policy.env >/dev/null && echo "pg61 tree present"
grep -E '^DS41_(PORT|SERVED|KV|MAXLEN|OFFLOAD_GB|GPU_MODE)=' $PG61_WORK_REAL/policy.env | tr '\n' ' '; echo
echo "coldskip,peer3 count: $(grep -c ',coldskip,peer3,' $PG61_WORK_REAL/policy.env) (want 1)"
echo "DEC HEAD: $(git -C $DEC rev-parse --short HEAD) (want fadc3d2)   shards: $(ls $MODEL_DIR/model-*.safetensors 2>/dev/null | wc -l) (want >= 48)"
free -g | sed -n 2p; echo "swap: $(swapon --show | wc -l) lines   THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
echo "== step 2: CDI pass-through by UUID"
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
echo "GB300 uuid: $U"
docker run --rm --device nvidia.com/gpu=$U --entrypoint nvidia-smi vllm/vllm-openai:v0.31.0 -L 2>&1 | tail -2 | cut -c1-200
echo "cdi specs: $(ls /etc/cdi /var/run/cdi 2>/dev/null | tr '\n' ' ')"
echo "== step 4: layout"
mkdir -p $GR/launcher $GR/promote $GR/home
ln -sfn $PG61_WORK_REAL $GR/pg61-work
ln -sfn /raid/work/src/garner $GR/src-git
ln -sfn $JM/deepseek-v4.1-flash-vllm-uva-dspark-sidecar/scripts/launch-pg61.sh $GR/launcher/launch-pg61.sh
printf '#!/bin/bash\necho "no Pg61 lane on this box: nothing to restore"\nexit 0\n' > $GR/promote/restore-pg61-after-gt-mt-1108.sh
chmod +x $GR/promote/restore-pg61-after-gt-mt-1108.sh
ls -l $GR | tail -n +2 | awk '{print $9, $10, $11}' | tr '\n' ';'; echo
echo "== step 5: patched private copy of the card"
cd $GTROOT/card
cp $JM/deepseek-v4.1-flash-vllm-gracetier/scripts/card-gt1.sh card-gt1.sh
if git apply --check card-gt1-station.diff 2>&1; then patch -p1 < card-gt1-station.diff; else echo "PATCH CHECK FAILED"; fi
bash -n card-gt1.sh && echo "syntax ok; override names present: $(grep -c 'GT1_KEEP\|GT1_SCRATCH\|GT1_LANE_DIR\|GT1_A5_GAUNTLET' card-gt1.sh) lines (want 5)"
diff <(cat $JM/deepseek-v4.1-flash-vllm-gracetier/scripts/card-gt1.sh) card-gt1.sh | grep -c '^[<>]' | sed 's/^/changed lines vs source: /'
