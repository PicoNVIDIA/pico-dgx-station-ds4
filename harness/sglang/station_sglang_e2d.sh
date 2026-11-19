#!/bin/bash
# Patched SGLang (Marlin SM gate removed, NOT stock), shape A again, only to re-time the stream windows.
# The second pass timed them from SGLang's generation counter, which moves only when a request FINISHES, so its
# window numbers under-count. This pass counts tokens at the client as they arrive (bench_sgl.py v2) and keeps
# the server counter beside them. Runs in tmux "sgl4". Log ends with DONE.
set -u
TS=$(date -u +%Y%m%dT%H%M%SZ); Q=/raid/work/logs/sgl4-$TS; mkdir -p "$Q"; ln -sfn "$Q" /raid/work/logs/sgl4-latest
IMG=lmsysorg/sglang:nightly-dev-cu13-20261119-6b374a55@sha256:cda9ed841f850ee6eb1892b50264af9c76de3add8dd476c2ebc0dbf4c0b3f62c
MODEL=/raid/models/DeepSeek-V4.1-Flash-1017
PORT=30010; NAME=dsv41-sglang
ALL=fixed,gsm8k_c1,greedy_c1,coding,streams,prefill,capacity,quality
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
free_gib() { free -g | awk '/^Mem:/{print "host used " $3 " GiB, free " $4 " GiB"}'; }
stage() {  # label, stages to run (comma list), cap minutes, extra args
  local label=$1 run=$2 cap=$3; shift 3
  local skip; skip=$(echo "$ALL" | tr ',' '\n' | grep -vxF -f <(echo "$run" | tr ',' '\n') | paste -sd, -)
  echo "-- $label [$run] $(date -u +%T)"
  timeout $(( cap * 60 )) python3 -I /raid/work/run/bench_sgl.py --base-url "http://127.0.0.1:$PORT" --model "$NAME" --key-file /raid/work/run/no-key \
    --out "$Q/$label.json" --skip "$skip" "$@" > "$Q/$label.log" 2>&1
  echo "$label: rc=$?"
  python3 /raid/work/run/bench_sum.py "$label" "$Q/$label.json"
  docker logs "$NAME" 2>&1 | grep -E "Decode batch" | tail -n 1 | sed -E 's/^.*Decode batch/  server: Decode batch/' | cut -c1-260
}
{
  date -u +'start %FT%TZ'
  docker ps --format '{{.Names}}' | grep -q . && { echo "a container is running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; echo DONE; exit 0; }
  docker rm -f "$NAME" > /dev/null 2>&1
  docker run -d --name "$NAME" -p "$PORT:$PORT" --gpus all --shm-size 32g --ipc=host --memory 484g -v "$MODEL":/model:ro \
    -v /raid/cache/sglang:/root/.cache -e HF_HUB_OFFLINE=1 -e SGLANG_CACHE_DIR=/root/.cache/sglang \
    -v /raid/work/run/sglang_e2_entry.sh:/entry.sh:ro "$IMG" bash /entry.sh \
    --model-path /model --served-model-name "$NAME" --host 0.0.0.0 --port "$PORT" --trust-remote-code --enable-metrics \
    --enable-paged-experts --paged-experts-num-resident 255 --moe-runner-backend marlin --disable-shared-experts-fusion \
    --engram-config '{"placement": "cpu"}' --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.90 --context-length 262144 \
    --cuda-graph-backend-prefill disabled --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 3600 \
    --max-running-requests 32 --cuda-graph-max-bs-decode 32 > /dev/null
  t0=$(date +%s); up=0
  while [ $(( $(date +%s) - t0 )) -lt 2700 ]; do
    [ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)" = running ] || break
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
    sleep 15
  done
  echo "boot: up=$up after $(( $(date +%s) - t0 )) s; gpu $(gpu_mib) MiB; $(free_gib)"
  if [ $up = 1 ]; then
    stage w-fixed fixed 15 --reps 3
    stage w-c1 gsm8k_c1,greedy_c1 10
    stage w-streams streams 20 --conc 2,4,8,16,32
    stage w-streams-r2 streams 20 --conc 2,4,8,16,32
  else
    docker logs "$NAME" 2>&1 | tail -n 5 | cut -c1-240
  fi
  docker logs "$NAME" > "$Q/server.log" 2>&1
  docker rm -f "$NAME" > /dev/null 2>&1; sleep 10; echo "after cleanup: gpu $(gpu_mib) MiB; $(free_gib)"
  date -u +'end %FT%TZ'
  echo DONE
} >> "$Q/run.log" 2>&1
