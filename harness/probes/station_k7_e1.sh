#!/bin/bash
# K7 step E1 on the Station (tmux "k7e1"): is moving a constant from the weight scale codes to the activation scale
# codes bit-exact in stock DeepGEMM MegaMoE? (research/K7-packed-scales.md, variant a2.) No kernel change.
# The probe is UNTESTED code. It runs in the mini-test container of bench/box/gracetier_minitest.sh (image v0.31.0,
# the gracetier checkout at /work, the checkpoint at /ref) and needs the GPU idle: this script stops our engine.
set -u
WAIT_FOR=${1:-/raid/work/logs/gtq15-latest/run.log}
DEC=/raid/work/src/gb300-dsv41-decode; GT=/raid/work/gt/gracetier; M=/raid/models/DeepSeek-V4.1-Flash-1017
S=/raid/work/k7/scratch; J=/raid/cache/k7-jit; L=/raid/work/logs/k7-e1.log
mkdir -p "$S" "$J/tilelang" "$J/triton" "$J/nv" "$J/dg"
{
  date -u +'start %FT%TZ'
  until grep -q '^DONE$' "$WAIT_FOR" 2>/dev/null; do sleep 20; done
  ( cd "$GT" && WORK=$PWD DS41_HOME=$PWD/home bash launch-gt1.sh down ) > /dev/null 2>&1
  docker rm -f ds41 > /dev/null 2>&1
  for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 3000 ] && break; sleep 2; done
  echo "gpu idle: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) at $(date -u +%T)"
  cp /raid/work/run/patches/K7/k7_e1_sf_bias_probe.py "$DEC/bench/"
  for layer in ${K7_E1_LAYERS:-10}; do
    echo "== E1 layer $layer $(date -u +%T)"
    timeout 1500 docker run --rm --name k7e1 --network none --gpus all --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
      -v "$DEC":/work -w /work -v "$M":/ref:ro -v "$S":/work/scratch -v "$GT/tree/fastpath/hook":/hook:ro \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_MODEL_DIR=/ref \
      -v "$J/tilelang":/root/.tilelang -v "$J/triton":/root/.triton -v "$J/nv":/root/.nv -v "$J/dg":/root/.deep_gemm \
      -e DG_JIT_CACHE_DIR=/root/.deep_gemm --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      bench/k7_e1_sf_bias_probe.py --layer "$layer" --model-dir /ref --rowmap /hook/rowmap-mix-v1-h255.json --json "scratch/k7-e1-L$layer.json"
    echo "E1 layer $layer rc=$? (0 = every comparison exact, 4 = at least one differs, 2 = could not run)"
  done
  docker rm -f k7e1 > /dev/null 2>&1
  date -u +'end %FT%TZ'
  echo DONE
} >> "$L" 2>&1
