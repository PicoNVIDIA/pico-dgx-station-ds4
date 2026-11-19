#!/bin/bash
# Station prep for the DeepSeek build: dirs, uv, checkpoint download and image pull in tmux, repo clones.
# Idempotent: safe to re-run. Nothing here needs a token (public model, public images).
set -u
REV=08c8657f02291e11b6f6b63e0e16014f8e748518
M=/raid/models/DeepSeek-V4.1-Flash-1017
sudo -n mkdir -p /raid/models /raid/work/logs /raid/work/src /raid/cache
sudo -n chown -R "$USER":"$USER" /raid/models /raid/work /raid/cache
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/raid/work/logs/uv-install.log 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
echo "uv: $(uv --version 2>&1)"

if tmux has-session -t dl 2>/dev/null; then
  echo "download session already running"
else
  tmux new-session -d -s dl "export PATH=$HOME/.local/bin:\$PATH HF_HOME=/raid/cache/hf HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1; date -u +'start %FT%TZ' > /raid/work/logs/dl.log; uvx --from 'huggingface_hub[hf_xet]' hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision $REV --local-dir $M --max-workers 16 >> /raid/work/logs/dl.log 2>&1; echo \"exit=\$? \$(date -u +%FT%TZ)\" >> /raid/work/logs/dl.log"
  echo "download started in tmux session dl -> $M"
fi

if tmux has-session -t img 2>/dev/null; then
  echo "image pull session already running"
else
  tmux new-session -d -s img "docker pull vllm/vllm-openai:v0.31.0 > /raid/work/logs/img.log 2>&1; echo \"exit=\$? \$(date -u +%FT%TZ)\" >> /raid/work/logs/img.log"
  echo "image pull started in tmux session img"
fi

cd /raid/work/src
clone() { # url dir [commit]
  if [ -d "$2/.git" ]; then echo "have $2 @ $(git -C "$2" rev-parse --short HEAD)"; return; fi
  if git clone -q "$1" "$2" 2>/raid/work/logs/clone-$2.log; then
    [ -n "${3:-}" ] && git -C "$2" checkout -q "$3" 2>>/raid/work/logs/clone-$2.log
    echo "cloned $2 @ $(git -C "$2" rev-parse --short HEAD)"
  else
    echo "CLONE FAILED $2: $(tail -1 /raid/work/logs/clone-$2.log)"
  fi
}
clone https://github.com/garnermccloud/dgx-station-recipes.git garner 8043a6811a5feda70fdfb6e827eebb2ec25c65d1
clone https://github.com/J-M-Recipes/recipes.git jm-recipes 486221ebccfc831c56dd979c808948faba652fb9
clone https://github.com/jmeadlock/gb300-dsv41-decode.git gb300-dsv41-decode fadc3d2eb7289189d82ba40cbbdc9b3f10b357a2
clone https://github.com/tomascupr/dgx-station-recipes.git tomascupr 3f25e5fff74c4428179ce49efb0f7c51822dfb18
sleep 20
echo "--- after 20 s"; du -sh "$M" 2>/dev/null; tail -n 3 /raid/work/logs/dl.log | cut -c1-200; tail -n 2 /raid/work/logs/img.log 2>/dev/null | cut -c1-160
