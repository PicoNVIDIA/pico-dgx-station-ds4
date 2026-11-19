#!/bin/bash
# Stage 0b, part 3: first gracetier boot on the card-less Station (R2 runbook steps 3 and 6).
# Stops the stock baseline (its autotune cache stays in /raid/cache/dsv41-stock), then runs the patched card
# in tmux session "gt1": build $GT, boot, G1 gate, published A5 cells, leave serving (GT1_KEEP=1).
set -u
export GTROOT=/raid/work/gt
export GR=$GTROOT/garner-shim
export GT=$GTROOT/gracetier
export DEC=/raid/work/src/gb300-dsv41-decode
export MODEL_DIR=/raid/models/DeepSeek-V4.1-Flash-1017
export GT1_SCRATCH=/raid/work/logs/gt1
LOG=/raid/work/logs/gt1-card-$(date -u +%Y%m%dT%H%M%SZ).log
if tmux has-session -t gt1 2>/dev/null; then echo "session gt1 already exists"; exit 0; fi
echo "== stop baseline"
docker stop dsv41 >/dev/null 2>&1; sleep 5
echo "running containers: [$(docker ps --format '{{.Names}}' | tr '\n' ' ')]"
echo "gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)  compute apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
free -g | sed -n 2p
cat > /raid/work/run/gt_card_env.sh <<'EOF'
export GT1_WINDOW=station-first-boot GT1_RES=1 GT1_RES_CELLS=a5 GT1_SEED_CACHE=1
export GT1_RES_CONFIG='{"mode": "adaptive", "freeze": false, "hyst": 3.0}'
export GT1_RESTORE=never GT1_KEEP=1 GT1_A5_GAUNTLET=0 GT1_LANE_DIR=/raid/work/gt/lane
export GR=/raid/work/gt/garner-shim GT=/raid/work/gt/gracetier DEC=/raid/work/src/gb300-dsv41-decode
export MODEL_DIR=/raid/models/DeepSeek-V4.1-Flash-1017 GT1_SCRATCH=/raid/work/logs/gt1
EOF
ln -sfn "$LOG" /raid/work/logs/gt1-card-latest.log
tmux new-session -d -s gt1 "source /raid/work/run/gt_card_env.sh; cd $DEC && bash $GTROOT/card/card-gt1.sh > $LOG 2>&1; echo \"CARD_EXIT rc=\$? \$(date -u +%FT%TZ)\" >> $LOG"
sleep 25
echo "== card log so far"; cat "$LOG" | cut -c1-260 | tail -25
