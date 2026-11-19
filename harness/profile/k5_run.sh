#!/usr/bin/env bash
# k5_run.sh: one profiled cell of the gracetier server on the Station. Run ON THE BOX, by hand, one cell per boot.
#   bash k5_run.sh preflight            read-only checks (server may be up)
#   bash k5_run.sh cell TAG STREAMS     down -> patch -> nsys-wrapped up -> warm -> profiled cell -> export -> table
#   bash k5_run.sh restore              down -> un-patch -> plain up
# Env: GT=/raid/work/gt/gracetier  MODEL_DIR=/raid/models/DeepSeek-V4.1-Flash-1017  K5=<dir holding these scripts>
#      NSYS_DIR=/opt/nvidia/nsight-systems/<version>  (host install; mounted read-only into the container)
set -euo pipefail
GT=${GT:-/raid/work/gt/gracetier}; MODEL_DIR=${MODEL_DIR:-/raid/models/DeepSeek-V4.1-Flash-1017}
K5=${K5:-$(cd "$(dirname "$0")" && pwd)}; IMAGE=${IMAGE:-vllm/vllm-openai:v0.31.0}
NSYS_DIR=${NSYS_DIR:-$(ls -d /opt/nvidia/nsight-systems/*/ 2>/dev/null | sort -V | tail -1)}; NSYS_DIR=${NSYS_DIR%/}
NSYS_BIN=${NSYS_BIN:-$(command -v nsys || echo "$NSYS_DIR/bin/nsys")}
say(){ printf '%s [k5] %s\n' "$(date '+%F %T')" "$*"; }
up(){ ( cd "$GT" && WORK=$PWD DS41_HOME=$PWD/home MODEL_DIR=$MODEL_DIR "$@" bash launch-gt1.sh up ); }
down(){ ( cd "$GT" && WORK=$PWD DS41_HOME=$PWD/home bash launch-gt1.sh down ); }

case "${1:-}" in
preflight)
  say "host nsys: dir=${NSYS_DIR:-MISSING} bin=$NSYS_BIN"; "$NSYS_BIN" --version || say "nsys NOT on the host: see the K5 note, section 1.5"
  ls -d "$NSYS_DIR/target-linux-sbsa-armv8" || say "no sbsa target dir under $NSYS_DIR"
  say "image: profiler flags and nsys inside the image (CPU only, no GPU)"
  docker run --rm --network none --entrypoint bash "$IMAGE" -c '
    command -v nsys || echo "nsys: not in the image";
    V=/usr/local/lib/python3.12/dist-packages/vllm;
    grep -rn "VLLM_TORCH_PROFILER_DIR\|torch_profiler_dir\|ProfilerKind\|start_profile" $V/envs.py $V/config/profiler.py $V/entrypoints/serve/profile/*.py 2>/dev/null | cut -c1-200 | head -30;
    python3 -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda)"' || true
  ( cd "$GT" && patch -p1 -d tree --dry-run < "$K5/k5-serve-profile.patch" && say "patch dry-run OK" ) || say "patch does not apply (already applied?)"
  ;;
cell)
  TAG=${2:?tag}; N=${3:-1}; OUT=$GT/k5/$TAG; mkdir -p "$OUT"
  [ -x "$NSYS_DIR/target-linux-sbsa-armv8/nsys" ] || { say "no nsys target under $NSYS_DIR"; exit 5; }
  down || true
  grep -q nsyspfx "$GT/tree/serve/serve.sh" || { cp "$GT/tree/serve/serve.sh" "$GT/k5/serve.sh.pre-k5"; patch -p1 -d "$GT/tree" --no-backup-if-mismatch < "$K5/k5-serve-profile.patch"; }
  [ "$(grep -c nsyspfx "$GT/tree/serve/serve.sh")" = 3 ] || { say "unexpected patch result"; exit 6; }
  sudo rm -rf "$GT/home/state/prof/$TAG"
  up env DS41_NSYS_WRAP=1 DS41_NSYS_TAG="$TAG" DS41_NSYS_DIR="$NSYS_DIR" DS41_NSYS_GRAPH="${GRAPH:-node}"
  docker inspect -f '{{json .Args}}' ds41 | grep -q '/nsys","profile"' || { say "ds41 args carry no nsys wrap"; exit 7; }
  docker inspect -f '{{json .Args}}' ds41 > "$OUT/ds41-args.json"
  # warm under the wrap (collection not started): this is the same-boot untraced-range control
  python3 "$K5/k5_drive.py" "$OUT/warm.json" --streams "$N" --max-tokens 256 --warm "${WARM:-6}" | tee "$OUT/warm.log"
  # the profiled cell; a stall watchdog is the caller's job: if gen tokens do not move for 60 s, run: bash k5_run.sh restore
  timeout 600 python3 "$K5/k5_drive.py" "$OUT/cell.json" --streams "$N" --max-tokens "${TOK:-256}" --warm 0 --profile | tee "$OUT/cell.log" || true
  for i in $(seq 1 120); do docker ps --format '{{.Names}}' | grep -qx ds41 || break; sleep 5; done
  docker logs ds41 2>&1 | grep -iE "nsys|profil|Generating|report" | tail -20 > "$OUT/ds41-nsys-lines.log" || true
  PROF=$GT/home/state/prof/$TAG; rep=$(ls -t "$PROF"/*.nsys-rep 2>/dev/null | head -1)
  [ -n "$rep" ] || { say "no .nsys-rep under $PROF (stall? see section 1.6)"; exit 8; }
  sudo "$NSYS_BIN" export --type sqlite --force-overwrite=true -o "$OUT/$TAG.sqlite" "$rep" > "$OUT/export.log" 2>&1
  sudo "$NSYS_BIN" stats --report cuda_gpu_kern_sum --format csv "$rep" > "$OUT/kern-sum.csv" 2> "$OUT/stats.err" || true
  sudo chown -R "$(id -u):$(id -g)" "$OUT"
  python3 "$K5/k5_step_table.py" "$OUT/$TAG.sqlite" --top 80 --csv "$OUT/table.csv" --json "$OUT/table.json" | tee "$OUT/table.txt"
  say "done: $OUT/table.txt. The server has exited. Next: another cell, or: bash k5_run.sh restore"
  ;;
restore)
  down || true
  if grep -q nsyspfx "$GT/tree/serve/serve.sh"; then patch -R -p1 -d "$GT/tree" --no-backup-if-mismatch < "$K5/k5-serve-profile.patch"; fi
  if ( cd "$GT" && grep ' tree/serve/serve.sh$' BUILD-SHA256SUMS | sha256sum -c --quiet - ); then say "serve.sh matches BUILD-SHA256SUMS again"; else say "WARNING: serve.sh differs from BUILD-SHA256SUMS (compare with k5/serve.sh.pre-k5)"; fi
  up env
  ;;
*) sed -n '2,8p' "$0"; exit 2 ;;
esac
