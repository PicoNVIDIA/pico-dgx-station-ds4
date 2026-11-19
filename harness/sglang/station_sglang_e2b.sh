#!/bin/bash
# SGLang with the Marlin SM gate removed (NOT stock), second run, this time measured.
# The first run (station_sglang_probe.sh, E2) booted and answered, but Garner's bench_full.py needs vLLM's /metrics,
# so nothing was timed before it was torn down. This run adds --enable-metrics and uses bench_sgl.py (bench_full.py
# with SGLang's metric names; same prompts and stages).
#   A  32 running requests, decode graphs to batch 32: one stream, 2-32 streams, prefill, quality
#   B  only if A does not serve: the shape that booted (1 running request): one stream, prefill, quality
# Runs on the Station in tmux "sgl2" after queue 14. Log ends with DONE.
set -u
WAIT_FOR=${1:-/raid/work/logs/gtq14-latest/run.log}
TS=$(date -u +%Y%m%dT%H%M%SZ); Q=/raid/work/logs/sgl2-$TS; mkdir -p "$Q" /raid/cache/sglang
ln -sfn "$Q" /raid/work/logs/sgl2-latest
IMG=lmsysorg/sglang:nightly-dev-cu13-20261119-6b374a55@sha256:cda9ed841f850ee6eb1892b50264af9c76de3add8dd476c2ebc0dbf4c0b3f62c
MODEL=/raid/models/DeepSeek-V4.1-Flash-1017
PORT=30010; NAME=dsv41-sglang
ALL=fixed,gsm8k_c1,greedy_c1,coding,streams,prefill,capacity,quality
COMMON=(--gpus all --shm-size 32g --ipc=host --memory 484g -v "$MODEL":/model:ro -v /raid/cache/sglang:/root/.cache -e HF_HUB_OFFLINE=1
        -e SGLANG_CACHE_DIR=/root/.cache/sglang -v /raid/work/run/sglang_e2_entry.sh:/entry.sh:ro)
SRV=(--model-path /model --served-model-name "$NAME" --host 0.0.0.0 --port "$PORT" --trust-remote-code --enable-metrics
     --enable-paged-experts --paged-experts-num-resident 255 --moe-runner-backend marlin --disable-shared-experts-fusion
     --engram-config '{"placement": "cpu"}' --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.90 --context-length 262144
     --cuda-graph-backend-prefill disabled --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 3600)

gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
free_gib() { free -g | awk '/^Mem:/{print "host used " $3 " GiB, free " $4 " GiB"}'; }
clear_srv() { docker rm -f "$NAME" > /dev/null 2>&1; for i in $(seq 1 30); do [ "$(gpu_mib)" -lt 3000 ] && break; sleep 2; done; echo "  after cleanup: gpu $(gpu_mib) MiB; $(free_gib)"; }
wait_srv() {  # label, cap minutes -> 0 serving, 1 exited, 2 cap
  local label=$1 cap=$2 t0=$(date +%s) st
  while :; do
    st=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo gone)
    if [ "$st" != running ]; then echo "$label: container $st after $(( $(date +%s) - t0 )) s"; return 1; fi
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then echo "$label: HEALTH 200 after $(( $(date +%s) - t0 )) s"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -ge $(( cap * 60 )) ]; then echo "$label: CAP of $cap min reached"; return 2; fi
    sleep 15
  done
}
keep_log() {  # label
  docker logs "$NAME" > "$Q/$1.log" 2>&1
  grep -nE "RuntimeError|ValueError|OutOfMemory|out of memory|cudaHostRegister|Killed|not supported|sm_103 probe|Paged experts: (decode|K|resident|[0-9])|max_total_num_tokens|KV cache" "$Q/$1.log" | tail -n 12 | cut -c1-260
  echo "$1: last lines:"; tail -n 4 "$Q/$1.log" | cut -c1-240
}
smoke() {
  curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":16,\"temperature\":0}" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print("smoke answer:", repr(d["choices"][0]["message"]["content"])[:80])' 2>&1 | cut -c1-200
  echo "metrics: $(curl -s -m 10 "http://127.0.0.1:$PORT/metrics" | grep -c '^sglang:') sglang lines; generation counter: $(curl -s -m 10 "http://127.0.0.1:$PORT/metrics" | grep -m1 '^sglang:generation_tokens_total' | cut -c1-120)"
}
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
measure() {  # tag, with streams (1/0)
  local t=$1
  smoke
  stage "$t-fixed" fixed 25 --reps 5
  stage "$t-coding" coding 25 --coding-reps 5
  stage "$t-c1" gsm8k_c1,greedy_c1 10
  [ "$2" = 1 ] && stage "$t-streams" streams 20 --conc 2,4,8,16,32
  stage "$t-prefill" prefill 30
  stage "$t-quality" quality 75
  echo "$t: $(free_gib); gpu $(gpu_mib) MiB; restarts $(docker inspect -f '{{.RestartCount}}' "$NAME" 2>/dev/null)"
}

{
  date -u +'start %FT%TZ'; echo "waiting for DONE in $WAIT_FOR"
  until grep -q '^DONE$' "$WAIT_FOR" 2>/dev/null; do sleep 20; done
  date -u +'box free %T'
  ( cd /raid/work/gt/gracetier && WORK=$PWD DS41_HOME=$PWD/home bash launch-gt1.sh down ) > "$Q/engine-down.log" 2>&1
  docker rm -f ds41 > /dev/null 2>&1
  clear_srv
  grep -c 'sm_103 probe' /raid/work/run/sglang_e2_entry.sh > /dev/null || { echo "entry script missing"; echo DONE; exit 0; }

  echo "== A patched Paged Experts, 32 running requests $(date -u +%T)"
  docker run -d --name "$NAME" -p "$PORT:$PORT" "${COMMON[@]}" "$IMG" bash /entry.sh "${SRV[@]}" \
    --max-running-requests 32 --cuda-graph-max-bs-decode 32 > /dev/null
  wait_srv A 50; a=$?
  keep_log e2b-a
  if [ $a -eq 0 ]; then
    measure a 1; keep_log e2b-a-after
  else
    clear_srv
    echo "== B patched Paged Experts, 1 running request (the shape that booted) $(date -u +%T)"
    docker run -d --name "$NAME" -p "$PORT:$PORT" "${COMMON[@]}" "$IMG" bash /entry.sh "${SRV[@]}" \
      --max-running-requests 1 --cuda-graph-max-bs-decode 1 > /dev/null
    wait_srv B 50; b=$?
    keep_log e2b-b
    [ $b -eq 0 ] && { measure b 0; keep_log e2b-b-after; }
  fi
  clear_srv
  date -u +'end %FT%TZ'
  echo DONE
} >> "$Q/run.log" 2>&1
