#!/bin/bash
# Stage 0a measurement: the stock baseline on :8000, with the two public rulers.
#  1. Garner's bench_full.py exactly as Tomas Cupr ran his stock arm (published: 171.1 fixed prompt, 130.4 GSM8K,
#     326 at 8 streams, 584 at 32).
#  2. Meadlock's A5 cells (the ruler behind the 261 tok/s gracetier number): passes A and B, 8 and 32 streams.
# Runs detached in tmux session "sbench"; writes DONE when finished.
set -u
OUT=/raid/work/logs/stock-bench-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT" /raid/work/run
: > /raid/work/run/no-key
ln -sfn "$OUT" /raid/work/logs/stock-bench-latest
if tmux has-session -t sbench 2>/dev/null; then echo "session sbench already exists"; exit 0; fi
tmux new-session -d -s sbench "{ date -u +'start %FT%TZ'; \
  cd /raid/work/src/garner/deepseek-v4.1-flash/verify && python3 -I bench_full.py --base-url http://127.0.0.1:8000 --key-file /raid/work/run/no-key --out $OUT/bench-dsv41-gb300-alone.json --skip greedy_c1,coding,prefill,capacity,quality --conc 8,32 --reps 5; echo \"bench_full rc=\$?\"; \
  cd /raid/work/src/gb300-dsv41-decode && W0_BASE=http://127.0.0.1:8000 W0_MODEL=deepseek-v4.1-flash A5_OUT=$OUT/a5-stock A5_ARM=stock A5_WARM=1 A5_GAUNTLET=0 bash bench/box/a5-cells.sh; echo \"a5 rc=\$?\"; \
  date -u +'end %FT%TZ'; echo DONE; } > $OUT/run.log 2>&1"
echo "started tmux session sbench; output in $OUT"
