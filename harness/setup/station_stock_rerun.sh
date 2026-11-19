#!/bin/bash
# Stock vLLM v0.32.0 re-run (tmux "stock2"): the first stock session was one unwarmed run of four cells. This one
# warms the server with the same rounds our cells get, then takes the full scoreboard (every stage, including
# quality and capacity), two more stream passes, the two prompt sets twice, and a fresh identity capture.
# Stops our engine first and stops stock at the end. Ends its log with DONE.
set -u
OUT=/raid/work/logs/stock2-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; ln -sfn "$OUT" /raid/work/logs/stock2-latest
VERIFY=/raid/work/src/garner/deepseek-v4.1-flash/verify; DEC=/raid/work/src/gb300-dsv41-decode
B=http://127.0.0.1:8000; M=deepseek-v4.1-flash
{
  date -u +'start %FT%TZ'
  ( cd /raid/work/gt/gracetier && WORK=$PWD DS41_HOME=$PWD/home bash launch-gt1.sh down ) > "$OUT/engine-down.log" 2>&1
  docker rm -f ds41 ds41-sidecar > /dev/null 2>&1; sleep 8
  if docker ps --format '{{.Names}}' | grep -q .; then echo "a container is still running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; echo DONE; exit 0; fi
  if ! docker start dsv41 > "$OUT/start.log" 2>&1; then echo "docker start dsv41 failed: $(tail -n 1 "$OUT/start.log" | cut -c1-200)"; echo DONE; exit 0; fi
  t0=$(date +%s); up=0
  while [ $(( $(date +%s) - t0 )) -lt 6000 ]; do
    [ "$(docker inspect -f '{{.State.Status}}' dsv41 2>/dev/null)" = running ] || break
    if curl -sf -m 5 "$B/v1/models" | grep -q "$M"; then up=1; break; fi
    sleep 15
  done
  echo "stock: up=$up after $(( $(date +%s) - t0 )) s; $(docker logs dsv41 2>&1 | grep -m1 'GPU KV cache size' | sed -E 's/^.*GPU KV/GPU KV/' | cut -c1-140)"
  docker logs dsv41 2>&1 | grep -m1 'non-default args' | grep -oE "(max_num_seqs|gpu_memory_utilization|max_model_len|cpu_offload_gb|speculative_config)[^,]*,[^,]*" | tr '\n' ';' | cut -c1-400; echo
  if [ "$up" != 1 ]; then docker logs dsv41 2>&1 | tail -n 5 | cut -c1-300; docker stop dsv41 > /dev/null 2>&1; echo "STOCK NOT SERVING"; echo DONE; exit 0; fi
  bash /raid/work/run/smoke.sh 8000 "$M" 2>&1 | tail -n 1
  pass() { ( cd "$DEC" && W0_BASE=$B W0_MODEL=$M timeout 1800 python3 bench/gt_mix_drive.py pass --set "$2" --out "$OUT/$1.json" > "$OUT/$1.log" 2>&1 ); echo "$1: $(tail -n 1 "$OUT/$1.log" | sed -E 's/; per-prompt.*//; s/^pass [AB]: 20\/20 prompts, //' | cut -c1-170)"; }
  bf() {
    local label=$1 skip=$2; shift 2
    ( cd "$VERIFY" && python3 -I bench_full.py --base-url "$B" --key-file /raid/work/run/no-key --out "$OUT/$label.json" --skip "$skip" "$@" ) > "$OUT/$label.log" 2>&1
    local rc=$?
    grep -E '^\| ' "$OUT/$label.md" 2>/dev/null | grep -E '[0-9]' | grep -vE '^\| (shape|streams|prompt tokens) ' | sed -E "s/^/$label: /" | cut -c1-220
    grep -E 'preemptions|^\| tool call' "$OUT/$label.md" 2>/dev/null | sed -E "s/^/$label: /" | cut -c1-330
    [ $rc -eq 0 ] || { echo "$label: bench_full rc=$rc"; tail -n 2 "$OUT/$label.log" | cut -c1-200; }
  }
  for s in A B A B; do pass "stock-warm-$s" "$s" > /dev/null; done
  for r in 1 2; do pass "stock-c1-r$r-a" A; pass "stock-c1-r$r-b" B; done
  bf stock-score "" --conc 2,4,8,16,32 --reps 5 < /dev/null
  for r in 1 2; do bf "stock-streams-r$r" "fixed,gsm8k_c1,greedy_c1,coding,prefill,capacity,quality" --conc 2,4,8,16,32 < /dev/null; done
  ( cd /raid/work/run && python3 ident_capture.py "$B" "$M" /raid/work/ident/stock-v032-run3.json A,B 256 1 2>&1 | tail -n 1 | cut -c1-200
    python3 ident_cmp.py /raid/work/ident/stock-v032-run1.json /raid/work/ident/stock-v032-run3.json | sed -n 2p | sed 's/^/stock run3 against run1 (another boot): /' | cut -c1-200 )
  echo "server log error lines: $(docker logs dsv41 2>&1 | grep -cE ' ERROR |Traceback'); restarts $(docker inspect -f '{{.RestartCount}}' dsv41 2>/dev/null)"
  docker stop dsv41 > /dev/null 2>&1; sleep 8
  date -u +'end %FT%TZ'; echo DONE
} > "$OUT/run.log" 2>&1
