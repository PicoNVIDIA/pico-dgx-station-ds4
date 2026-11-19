#!/bin/bash
# Local test of the runner's held-out function (HO-BEGIN / HO-END in station_cells.sh) with stubbed tools.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=${TMPDIR:-/tmp}/ho-test; rm -rf "$T"; mkdir -p "$T/res"
Q=$T; RESDIR=$T/res; export Q RESDIR
printf '{"event": "drain", "cold": 500, "sel": 10000, "hot_moved": 7}\nnot json\n' > "$RESDIR/telemetry.jsonl"
sed -n '/^# HO-BEGIN/,/^# HO-END/p' "$HERE/station_cells.sh" > "$T/fn.sh"
ALL=fixed,gsm8k_c1,greedy_c1,coding,streams,prefill,capacity,quality
only() { echo "$ALL" | tr ',' '\n' | grep -vx "$1" | paste -sd, -; }
K=0
pass() { K=$((K + 1)); printf '{"event": "drain", "cold": %d, "sel": 10000, "hot_moved": %d}\n{"event": "other"}\n' $((400 - K * 50)) $((7 + K)) >> "$RESDIR/telemetry.jsonl"; echo "$1: errors 0; decode tok/s 3$K.0 (stub set $2)"; }
bf() { local label=$1; shift 2; printf '{"fixed": {"runs": [{"out_tok_s": 480.1}, {"out_tok_s": 520.2}, {"out_tok_s": 540.3}]}}' > "$Q/$label.json"; printf '{"event": "drain", "cold": 10, "sel": 10000, "hot_moved": 99}\n' >> "$RESDIR/telemetry.jsonl"; echo "$label: | fixed prompt | stub | args: $*"; }
# shellcheck disable=SC1090
source "$T/fn.sh"
out=$(heldout tcell 2>&1); rc=$?
echo "$out"
fail=0
chk() { if echo "$out" | grep -qE "$1"; then echo "PASS: $2"; else echo "FAIL: $2"; fail=1; fi; }
[ $rc -eq 0 ] && echo "PASS: function returned 0" || { echo "FAIL: rc $rc"; fail=1; }
chk "tcell-ho set A, first contact on the boot map: cold picks 350 of 10000 = 3.50 % over 1 drains" "phase 1 reads only its own lines"
chk "tcell-ho set A, passes 2 and 3: cold picks 550 of 20000 = 2.75 % over 2 drains" "phase 2 sums two passes"
chk "tcell-ho set B, first contact \(held out\): cold picks 200 of 10000 = 2.00 %" "held-out phase"
chk "tcell-ho-B-first: errors 0" "held-out pass printed"
chk "tcell-ho code prompt, never seen in this boot: cold picks 10 of 10000 = 0.10 % over 1 drains; hot experts moved since boot 99" "code prompt phase"
chk "tcell-ho code prompt runs, tok/s: 480.1, 520.2, 540.3" "per-run list printed"
chk "args: --reps 5" "fixed stage arguments"
[ "$(echo "$out" | grep -c 'cold picks')" = 6 ] && echo "PASS: six phases" || { echo "FAIL: phase count $(echo "$out" | grep -c 'cold picks')"; fail=1; }
[ $fail -eq 0 ] && echo "ALL OK" || echo "SOME FAILED"
exit $fail
