#!/usr/bin/env bash
set -euo pipefail
F=$(python3 -c 'import sglang, os; print(os.path.join(os.path.dirname(sglang.__file__), "srt/layers/quantization/mxfp4_marlin_moe.py"))')
grep -n 'MXFP4 Marlin requires SM90 or SM120' "$F"
sed -i 's/raise RuntimeError("MXFP4 Marlin requires SM90 or SM120.")/pass  # sm_103 probe: gate removed/' "$F"
grep -n 'sm_103 probe' "$F"
exec python3 -m sglang.launch_server "$@"
