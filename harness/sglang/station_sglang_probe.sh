#!/bin/bash
# SGLang on one GB300: box evidence for research/S1-sglang-one-gb300.md. Runs on the Station in tmux "sgl".
# Waits for queue 13, stops our engine, then in series (one GPU):
#   P0  megakernel at TP1                 expected: refused at argument resolution
#   P1  stock Paged Experts, Marlin       expected: RuntimeError "MXFP4 Marlin requires SM90 or SM120."
#   E2  P1 with the SM gate removed       NOT stock SGLang. If it serves: smoke, then bench_full at one stream
#   F3  stock --cpu-offload-gb 100        expected: fails or crawls; closes "does any stock flag set load"
# Every step has a time cap. Log ends with DONE. Nothing here touches our engine tree.
set -u
WAIT_FOR=${1:-/raid/work/logs/gtq13-latest/run.log}
TS=$(date -u +%Y%m%dT%H%M%SZ); Q=/raid/work/logs/sgl-$TS; mkdir -p "$Q" /raid/cache/sglang
ln -sfn "$Q" /raid/work/logs/sgl-latest
IMG=lmsysorg/sglang:nightly-dev-cu13-20261119-6b374a55@sha256:cda9ed841f850ee6eb1892b50264af9c76de3add8dd476c2ebc0dbf4c0b3f62c
MODEL=/raid/models/DeepSeek-V4.1-Flash-1017
VERIFY=/raid/work/src/garner/deepseek-v4.1-flash/verify
PORT=30010; NAME=dsv41-sglang
COMMON=(--gpus all --shm-size 32g --ipc=host -v "$MODEL":/model:ro -v /raid/cache/sglang:/root/.cache -e HF_HUB_OFFLINE=1)
SRV=(--model-path /model --served-model-name "$NAME" --host 0.0.0.0 --port "$PORT" --trust-remote-code)
PAGED=(--enable-paged-experts --paged-experts-num-resident 255 --moe-runner-backend marlin --disable-shared-experts-fusion
       --engram-config '{"placement": "cpu"}' --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.90
       --max-running-requests 1 --cuda-graph-max-bs-decode 1 --cuda-graph-backend-prefill disabled
       --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 3600)

gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
free_gib() { free -g | awk '/^Mem:/{print "host used " $3 " GiB, free " $4 " GiB, cache " $6 " GiB"}'; }
clear_srv() { docker rm -f "$NAME" > /dev/null 2>&1; for i in $(seq 1 30); do [ "$(gpu_mib)" -lt 3000 ] && break; sleep 2; done; echo "  after cleanup: gpu $(gpu_mib) MiB; $(free_gib)"; }
# wait_srv LABEL CAP_MINUTES -> 0 serving, 1 exited, 2 cap reached
wait_srv() {
  local label=$1 cap=$2 t0=$(date +%s) st
  while :; do
    st=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo gone)
    if [ "$st" != running ]; then echo "$label: container $st after $(( $(date +%s) - t0 )) s, exit code $(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null)"; return 1; fi
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then echo "$label: HEALTH 200 after $(( $(date +%s) - t0 )) s"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -ge $(( cap * 60 )) ]; then echo "$label: CAP of $cap min reached, still loading"; return 2; fi
    sleep 15
  done
}
keep_log() {  # label
  docker logs "$NAME" > "$Q/$1.log" 2>&1
  echo "$1: log $(wc -l < "$Q/$1.log") lines; error lines:"
  grep -nE "RuntimeError|ValueError|OutOfMemory|out of memory|Paged experts|MXFP4 Marlin|cudaHostRegister|Killed|Traceback|not supported|sm_103 probe|huge pages" "$Q/$1.log" | tail -n 14 | cut -c1-260
  echo "$1: last lines:"; tail -n 6 "$Q/$1.log" | cut -c1-260
}
smoke() {
  curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":16,\"temperature\":0}" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print("smoke answer:", repr(d["choices"][0]["message"]["content"])[:80])' 2>&1 | cut -c1-200
}
bench() {  # label, skip list, cap minutes, extra args
  local label=$1 skip=$2 cap=$3; shift 3
  ( cd "$VERIFY" && timeout $(( cap * 60 )) python3 -I bench_full.py --base-url "http://127.0.0.1:$PORT" --model "$NAME" --key-file /raid/work/run/no-key --out "$Q/$label.json" --skip "$skip" "$@" ) > "$Q/$label.log" 2>&1
  echo "$label: bench_full rc=$?"
  grep -E '^\| ' "$Q/$label.md" 2>/dev/null | grep -E '[0-9]' | grep -vE '^\| (shape|streams|prompt tokens) ' | sed -E "s/^/$label: /" | cut -c1-220
  grep -E '^\| tool call' "$Q/$label.md" 2>/dev/null | sed -E "s/^/$label: /" | cut -c1-200
  tail -n 2 "$Q/$label.log" | cut -c1-240
}

{
  date -u +'start %FT%TZ'; echo "waiting for DONE in $WAIT_FOR"
  until grep -q '^DONE$' "$WAIT_FOR" 2>/dev/null; do sleep 20; done
  date -u +'box free %T'
  ( cd /raid/work/gt/gracetier && WORK=$PWD DS41_HOME=$PWD/home bash launch-gt1.sh down ) > "$Q/engine-down.log" 2>&1
  docker rm -f ds41 > /dev/null 2>&1
  for i in $(seq 1 60); do [ "$(gpu_mib)" -lt 3000 ] && break; sleep 2; done
  echo "engine down: gpu $(gpu_mib) MiB; $(free_gib)"

  echo "== PULL $(date -u +%T)"
  docker pull "$IMG" > "$Q/pull.log" 2>&1; echo "pull rc=$? arch $(docker image inspect --format '{{.Architecture}}' "$IMG" 2>&1 | cut -c1-60) size $(docker image inspect --format '{{.Size}}' "$IMG" 2>/dev/null | awk '{printf "%.1f GB", $1/1e9}')"
  docker image inspect "$IMG" > /dev/null 2>&1 || { tail -n 3 "$Q/pull.log" | cut -c1-200; echo "NO IMAGE: stopping"; echo DONE; exit 0; }
  docker run --rm --gpus all --entrypoint python3 "$IMG" -c 'import sglang, torch; print("sglang", sglang.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))' 2>&1 | tail -n 2 | cut -c1-240

  echo "== P0 megakernel at TP1 $(date -u +%T)"
  timeout 900 docker run --rm --name "$NAME-p0" "${COMMON[@]}" -e SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample "$IMG" \
    python3 -m sglang.launch_server "${SRV[@]}" --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
    --kv-cache-dtype fp8_e4m3 --dsv4-attn-backend flashmla --enable-nvidia-megakernel > "$Q/p0.log" 2>&1
  echo "P0: rc=$? ($(wc -l < "$Q/p0.log") log lines)"
  grep -nE "cannot run this setup|tp_size|is not one of|ValueError|RuntimeError|DeepSeek-V4.1 on sm_103" "$Q/p0.log" | tail -n 10 | cut -c1-260
  docker rm -f "$NAME-p0" > /dev/null 2>&1

  echo "== P1 stock Paged Experts + Marlin $(date -u +%T)"
  clear_srv
  docker run -d --name "$NAME" -p "$PORT:$PORT" "${COMMON[@]}" -e SGLANG_CACHE_DIR=/root/.cache/sglang "$IMG" \
    python3 -m sglang.launch_server "${SRV[@]}" "${PAGED[@]}" --context-length 131072 > /dev/null
  wait_srv P1 30; p1=$?
  keep_log p1
  [ $p1 -eq 0 ] && { echo "P1 SERVES (the verdict was wrong): keeping numbers"; smoke; bench p1-score streams,capacity 60 --reps 5; }
  clear_srv

  echo "== E2 Paged Experts + Marlin with the SM gate removed (NOT stock) $(date -u +%T)"
  cat > /raid/work/run/sglang_e2_entry.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
F=$(python3 -c 'import sglang, os; print(os.path.join(os.path.dirname(sglang.__file__), "srt/layers/quantization/mxfp4_marlin_moe.py"))')
grep -n 'MXFP4 Marlin requires SM90 or SM120' "$F"
sed -i 's/raise RuntimeError("MXFP4 Marlin requires SM90 or SM120.")/pass  # sm_103 probe: gate removed/' "$F"
grep -n 'sm_103 probe' "$F"
exec python3 -m sglang.launch_server "$@"
EOS
  docker run -d --name "$NAME" -p "$PORT:$PORT" --memory 484g "${COMMON[@]}" -e SGLANG_CACHE_DIR=/root/.cache/sglang \
    -v /raid/work/run/sglang_e2_entry.sh:/entry.sh:ro "$IMG" \
    bash /entry.sh "${SRV[@]}" "${PAGED[@]}" --context-length 262144 > /dev/null
  wait_srv E2 75; e2=$?
  keep_log e2
  if [ $e2 -eq 0 ]; then
    echo "E2 SERVES: $(free_gib); gpu $(gpu_mib) MiB"; smoke
    bench e2-fixed gsm8k_c1,greedy_c1,coding,streams,prefill,capacity,quality 20 --reps 5
    bench e2-score streams,capacity 75 --reps 5
    keep_log e2-after
  fi
  clear_srv

  echo "== F3 stock generic offload (--cpu-offload-gb 100) $(date -u +%T)"
  docker run -d --name "$NAME" -p "$PORT:$PORT" --memory 484g "${COMMON[@]}" "$IMG" \
    python3 -m sglang.launch_server "${SRV[@]}" --engram-config '{"placement": "cpu"}' --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 --context-length 131072 --max-running-requests 1 --cuda-graph-max-bs-decode 1 \
    --cuda-graph-backend-prefill disabled --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 3600 \
    --cpu-offload-gb 100 > /dev/null
  wait_srv F3 30; f3=$?
  keep_log f3
  [ $f3 -eq 0 ] && { echo "F3 SERVES: $(free_gib); gpu $(gpu_mib) MiB"; smoke; bench f3-fixed gsm8k_c1,greedy_c1,coding,streams,prefill,capacity,quality 20 --reps 3; }
  clear_srv

  date -u +'end %FT%TZ'
  echo DONE
} >> "$Q/run.log" 2>&1
