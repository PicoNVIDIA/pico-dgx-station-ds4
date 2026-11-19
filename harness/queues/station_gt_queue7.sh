#!/bin/bash
# Queue 7 on the Station (tmux "gtq7"): takes over from queue 6 after its h292u98 cell, runs the K6 audit cell,
# and only if the audit is clean (identity 40/40, skips happened, every one audited, zero differing words, no error
# line) goes on to the measured K6 cells (queue name gtq7b).
set -u
L6=$(readlink -f /raid/work/logs/gtq6-latest)/run.log
until grep -qE '^(CELL_DONE h292u98|DONE)' "$L6" 2>/dev/null; do sleep 15; done
tmux kill-session -t gtq6 2>/dev/null
grep -q '^DONE$' "$L6" || printf 'CUT after h292u98 by queue 7 at %s (ord3, k1f7h, k1f3h postponed)\nDONE\n' "$(date -u +%T)" >> "$L6"
sleep 3
bash /raid/work/run/station_cells3.sh gtq7 /raid/work/run/cells/q7a.list
R=$(readlink -f /raid/work/logs/gtq7-latest)/run.log
G=/raid/work/logs/gtq7-gate.log
{
  date -u +'gate at %T'
  line=$(grep -E "K6 stage skip: steps" "$R" | tail -n 1)
  echo "last counter line: $line"
  skipped=$(echo "$line" | grep -oE "skipped [0-9]+" | grep -oE "[0-9]+")
  audited=$(echo "$line" | grep -oE "audited [0-9]+" | grep -oE "[0-9]+")
  bad=$(echo "$line" | grep -oE "mismatch_words [0-9]+" | grep -oE "[0-9]+")
  enabled=$(echo "$line" | grep -oE "enabled [0-9]+" | grep -oE "[0-9]+")
  errs=$(grep -oE "K6 error lines: [0-9]+" "$R" | tail -n 1 | grep -oE "[0-9]+$")
  ident=$(grep -c "k6a-c1 vs reference: identical outputs: 40/40" "$R")
  echo "skipped=${skipped:-?} audited=${audited:-?} mismatch_words=${bad:-?} enabled=${enabled:-?} error_lines=${errs:-?} identity_40_of_40=$ident"
  if [ "${skipped:-0}" -gt 0 ] && [ "${audited:-0}" -gt 0 ] && [ "${bad:-1}" = 0 ] && [ "${enabled:-0}" = 1 ] && [ "${errs:-1}" = 0 ] && [ "$ident" = 1 ]; then
    echo "GATE PASSED"
  else
    echo "GATE FAILED"
  fi
} > "$G" 2>&1
grep -q "GATE PASSED" "$G" || exit 1
exec bash /raid/work/run/station_cells3.sh gtq7b /raid/work/run/cells/q7b.list
