#!/bin/bash
# Stage 0a: boot the published stock baseline (vLLM's own DGX Station config, as scripted by Tomas Cupr)
# unchanged, except that the first start runs the FlashInfer autotune (his comment says to drop the
# AUTOTUNE_SKIP line for that). Runs detached in tmux session "stock": image pull, then the server.
set -u
mkdir -p /raid/work/run /raid/cache/dsv41-stock
SRC=/raid/work/src/tomascupr/glm-5.3-flash-fp8/baseline/serve-deepseek-gb300-stock.sh
RUN=/raid/work/run/serve-stock-first.sh
grep -v "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS" "$SRC" > "$RUN"
echo "lines: src $(wc -l < "$SRC") run $(wc -l < "$RUN")  (expect one fewer)"
diff <(grep -v "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS" "$SRC") "$RUN" >/dev/null && echo "run script = source minus the skip line"
if tmux has-session -t stock 2>/dev/null; then echo "session stock already exists"; exit 0; fi
tmux new-session -d -s stock "{ date -u +'start %FT%TZ'; docker pull vllm/vllm-openai:v0.32.0 2>&1 | tail -2; echo \"pull exit=\$?\"; sudo -n env DS41_MODEL_DIR=/raid/models/DeepSeek-V4.1-Flash-1017 DSV41_CACHE=/raid/cache/dsv41-stock sh $RUN; echo \"launch exit=\$? \$(date -u +%FT%TZ)\"; } > /raid/work/logs/stock-launch.log 2>&1"
echo "started tmux session stock; log /raid/work/logs/stock-launch.log"
