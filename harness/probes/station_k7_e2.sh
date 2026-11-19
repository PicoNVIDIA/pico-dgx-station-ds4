#!/bin/bash
# K7 step E2 on the Station (tmux "k7e2"): what does a packed scale column look like in shared memory after a TMA
# load, and in tensor memory after tcgen05.cp? (research/K7-E2-design.md.) One small test kernel, compiled inside
# the container with nvcc, about 2 minutes, under 1 GiB of GPU memory.
#
# Unlike station_k7_e1.sh this script takes no argument, waits for nothing and stops nothing. If ANY container is
# running it refuses and writes why. It needs neither the checkpoint nor the gracetier checkout: only the image
# and the three files k7_e2_probe.py, k7_e2_kernel.cu, k7_e2_ptx.cuh in $SRC.
# The probe is UNTESTED on a GPU (written, compiled for sm_103a and logic-tested on a Mac).
#
#   tmux new -d -s k7e2 'bash /raid/work/run/station_k7_e2.sh'      then: tail -f /raid/work/logs/k7-e2.log
set -u
SRC=${K7_SRC:-/raid/work/run/patches/K7}
S=/raid/work/k7/scratch; J=/raid/cache/k7-jit; L=/raid/work/logs/k7-e2.log
IMAGE=vllm/vllm-openai:v0.31.0
CAP=${K7_E2_CAP:-600}                       # seconds, the whole container
# K7_E2_ARGS: extra probe arguments, e.g. K7_E2_ARGS="--nvcc /usr/local/cuda/bin/nvcc" or "--skip-mma"
B=k7-e2-build-$(date -u +%Y%m%d-%H%M%S)     # a fresh build directory per run (the container writes as root)
mkdir -p "$S" "$J/nv" "$(dirname "$L")"
{
  date -u +'start %FT%TZ'
  running=$(docker ps --format '{{.Names}}' | tr '\n' ' ')
  missing=""
  for f in k7_e2_probe.py k7_e2_kernel.cu k7_e2_ptx.cuh; do [ -f "$SRC/$f" ] || missing="$missing $f"; done
  if [ -n "$running" ]; then
    echo "REFUSED: a container is running ($running). E2 stops nothing: run it when the GPU is free."
    echo "E2 rc=9 (not started)"
  elif [ -n "$missing" ]; then
    echo "REFUSED: missing in $SRC:$missing"
    echo "E2 rc=9 (not started)"
  elif ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "REFUSED: image $IMAGE is not on this host (E2 does not pull)"
    echo "E2 rc=9 (not started)"
  else
    echo "gpu before: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -1)"
    echo "sources: $(cd "$SRC" && sha256sum k7_e2_probe.py k7_e2_kernel.cu k7_e2_ptx.cuh | cut -c1-12,65- | tr '\n' ' ')"
    timeout "$CAP" docker run --rm --name k7e2 --network none --gpus all --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
      -v "$SRC":/k7:ro -v "$S":/scratch -w /scratch -v "$J/nv":/root/.nv \
      --entrypoint python3 "$IMAGE" \
      /k7/k7_e2_probe.py --build-dir "/scratch/$B" --json /scratch/k7-e2.json ${K7_E2_ARGS:-}
    rc=$?
    echo "E2 rc=$rc (0 = ran and classified, 2 = could not compile or run, 3 = ran but a control failed, 124 = time cap of $CAP s)"
    docker rm -f k7e2 > /dev/null 2>&1
    echo "raw results: $S/k7-e2.json   build logs: $S/$B"
    echo "gpu after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
  fi
  date -u +'end %FT%TZ'
  echo DONE
} >> "$L" 2>&1
