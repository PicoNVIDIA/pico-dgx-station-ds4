#!/bin/bash
Q=$1
BENCH_SGL_VLLM_NAMES=1 python3 -I /raid/work/run/bench_sgl.py --base-url http://127.0.0.1:30006 --model dsv41-flash-gracetier \
  --key-file /raid/work/run/no-key --out "$Q/rulerchk-client.json" --skip fixed,coding,prefill,capacity,quality --conc 2,8,32 > "$Q/rulerchk-client.log" 2>&1
echo "client-counted windows rc=$?"
python3 /raid/work/run/bench_sum.py rulerchk-client "$Q/rulerchk-client.json"
bash /raid/work/run/cells/k6_post.sh | tail -n 2
