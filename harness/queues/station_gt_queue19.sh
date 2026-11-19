#!/bin/bash
# Queue 19 on the Station (tmux "gtq19"): waits for queue 18, checks its four K7 gates in the log, and runs the K7
# speed cells only if every gate passed. The verdict goes to /raid/work/logs/gtq19-gate.log.
set -u
L18=$(readlink -f /raid/work/logs/gtq18-latest)/run.log; G=/raid/work/logs/gtq19-gate.log
until grep -q '^DONE$' "$L18" 2>/dev/null; do sleep 20; done
sec() { sed -n "/== CELL $1 /,/CELL_DONE $1/p" "$L18"; }
num() { echo "$1" | sed -nE "s/.*$2 ([0-9]+).*/\1/p"; }
fail=""
grep -q "k7ctlA-c1 vs reference: identical outputs: 40/40" "$L18" || fail="$fail; control A output not 40/40"
grep -q "k7ctlB-c1 vs reference: identical outputs: 40/40" "$L18" || fail="$fail; kernel-only control output not 40/40"
grep -q "k7audit-c1 vs reference: identical outputs: 40/40" "$L18" || fail="$fail; audit boot output not 40/40"
a=$(sec k7audit | grep "K7 packed scales: steps" | tail -n 1)
x=$(sec k7auditx | grep "K7 packed scales: steps" | tail -n 1)
[ -n "$a" ] || fail="$fail; no audit counters"
[ "$(num "$a" mismatch_words)" = 0 ] || fail="$fail; audit mismatch_words $(num "$a" mismatch_words)"
[ "$(num "$a" control_words)" = 0 ] || fail="$fail; audit control_words $(num "$a" control_words)"
[ "$(num "$a" kernel_err)" = 0 ] || fail="$fail; audit kernel_err $(num "$a" kernel_err)"
[ "$(num "$a" "audit calls")" -gt 0 ] 2>/dev/null || fail="$fail; no audit calls"
xm=$(num "$x" mismatch_words)
[ -n "$xm" ] && [ "$xm" -gt 0 ] || fail="$fail; the wrong-order control did not show mismatches (mismatch_words '${xm:-none}')"
{
  date -u +'%FT%TZ'
  echo "audit:        ${a#*K7 packed scales: }"
  echo "wrong order:  ${x#*K7 packed scales: }"
  if [ -z "$fail" ]; then echo "GATE PASSED: starting the K7 speed cells"; else echo "GATE FAILED:$fail"; echo "DONE"; fi
} > "$G" 2>&1
[ -z "$fail" ] || exit 0
exec bash /raid/work/run/station_cells5.sh gtq19 /raid/work/run/cells/q19.list
