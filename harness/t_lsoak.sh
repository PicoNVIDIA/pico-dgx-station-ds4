#!/bin/bash
# Local test of the runner's lsoak function with stubbed tools (no GPU, no server). Extracts the function between
# the LSOAK-BEGIN / LSOAK-END markers of station_cells.sh and runs 2 rounds with all sleeps shortened.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=${TMPDIR:-/tmp}/lsoak-test; rm -rf "$T"; mkdir -p "$T"
Q=$T; export Q
sed -n '/^# LSOAK-BEGIN/,/^# LSOAK-END/p' "$HERE/station_cells.sh" > "$T/fn.sh"
ALL=fixed,gsm8k_c1,greedy_c1,coding,streams,prefill,capacity,quality
only() { echo "$ALL" | tr ',' '\n' | grep -vx "$1" | paste -sd, -; }
sleep() { command sleep 0.2; }
N=0
bf() { local label=$1; shift 2; echo "{}" > "$Q/$label.json"; case "$label" in *lsoak-r*) command sleep 1; echo "$label: | 16 | 1,600 | stub | args: $*" ;; esac; }
pass() { echo "$1: errors 0; decode tok/s 320.0 (stub set $2)"; }
nvidia-smi() { N=$((N + 1)); echo $((250000 + RANDOM % 50)); }
free() { printf 'total used free shared buff/cache available\nMem: 506000 %d 1000 10 2000 %d\n' $((330000 + RANDOM % 100)) $((170000 - RANDOM % 100)); }
docker() { case "$1" in inspect) echo 0 ;; logs) echo "INFO ok" ;; esac; }
# shellcheck disable=SC1090
source "$T/fn.sh"
out=$(lsoak tcell 2 2>&1); rc=$?
echo "$out"
fail=0
chk() { if echo "$out" | grep -qE "$1"; then echo "PASS: $2"; else echo "FAIL: $2"; fail=1; fi; }
[ $rc -eq 0 ] && echo "PASS: function returned 0" || { echo "FAIL: rc $rc"; fail=1; }
chk "tcell-lsoak: round 1 of 2 done" "round 1 line"
chk "tcell-lsoak: round 2 of 2 done" "round 2 line"
chk "tcell-lsoak-r2: .*--conc 2,4,8,16,32 --window 600 --warmup 20" "stream call arguments"
chk "tcell-lsoak-r1-a: errors 0" "set A pass printed"
chk "prefill rounds [1-9][0-9]* with [1-9][0-9]* results" "prefill loop ran and wrote results"
chk "tcell-lsoak memory \([1-9][0-9]* samples\): GPU MiB first 2500[0-9][0-9] last 2500[0-9][0-9] min 2500[0-9][0-9] max 2500[0-9][0-9]; host used MiB first 3300[0-9][0-9]" "memory summary"
chk "server log error lines 0; restarts 0" "error and restart counts"
[ -f "$Q/tcell-lsoak.stop" ] && echo "PASS: stop flag written" || { echo "FAIL: no stop flag"; fail=1; }
sleep_left=$(jobs -r | wc -l | tr -d ' '); [ "$sleep_left" = 0 ] && echo "PASS: no background loop left running" || { echo "FAIL: $sleep_left background jobs still running"; fail=1; }
[ $fail -eq 0 ] && echo "ALL OK" || echo "SOME FAILED"
exit $fail
