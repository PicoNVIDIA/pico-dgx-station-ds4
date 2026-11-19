#!/bin/bash
# K7 step E0 on the Station (tmux "k7s"): CPU-only survey of every expert scale code in the checkpoint.
# Waits for queue 13 so it does not disturb one-stream timing cells; it overlaps the SGLang probes, which are not timed.
set -u
WAIT_FOR=${1:-/raid/work/logs/gtq13-latest/run.log}
S=/raid/work/run/patches/K7/k7_scale_survey.py; M=/raid/models/DeepSeek-V4.1-Flash-1017; L=/raid/work/logs/k7-survey.log
UV=$HOME/.local/bin/uv
{
  date -u +'start %FT%TZ'
  until grep -q '^DONE$' "$WAIT_FOR" 2>/dev/null; do sleep 20; done
  date -u +'== quick (5 layers) %T'
  nice -n 10 "$UV" run --with numpy python "$S" "$M" --quick --json /raid/work/logs/k7-survey-quick.json; echo "quick rc=$?"
  date -u +'== full (40 layers) %T'
  nice -n 10 "$UV" run --with numpy python "$S" "$M" --rowmap /raid/work/gt/gracetier/rowmap-allres.base.json --json /raid/work/logs/k7-survey.json; echo "full rc=$?"
  date -u +'end %FT%TZ'
  echo DONE
} >> "$L" 2>&1
