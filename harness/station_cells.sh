#!/bin/bash
# Generic cell runner for the Station (one boot per cell, one GPU, strictly in series).
# usage: station_cells.sh <queue-name> <cells-list>
#   cells-list lines:  name|profile      (lines starting with # are skipped)
#   /raid/work/run/cells/<name>.env      extra NAME=value lines appended to the base policy.env for that boot
#   /raid/work/run/cells/<name>.pre.sh   optional hook run before the boot (for example a new hot map)
# profile tokens, run in order:
#   c1           two measured rounds of Meadlock's sets A and B (one stream) + Garner's fixed code prompt x3
#   s:2,4,8xN    Garner's stream windows (60 s after a 20 s warm-up) at those stream counts, N repeats
#   pf           Garner's prefill stage (16K / 64K / 128K tokens, uncached, 3 runs each)
#   id           40-prompt greedy identity capture, compared with the reference capture
#   idc:8        the same 40 prompts sent 8 at a time, compared with the reference
#   res:FILE     send the live residency control document cells/FILE (JSON) and wait for the acknowledgement
#   soak         10 minutes at 16 streams while long uncached prompts arrive (see the case below)
#   lsoak:N      N rounds of about 53 minutes: 10-minute windows at 2, 4, 8, 16, 32 streams and sets A and B, with
#                long uncached prompts arriving throughout and GPU / host memory sampled every minute
#   ho           held-out placement test: set A three times, then set B on first contact and settled, then the code
#                prompt, with the cold share of expert picks after each phase. Give the cell an empty
#                cells/<name>.nowarm file so the usual warm-up rounds are skipped
# Every boot is warmed with two unmeasured rounds of sets A and B first. Ends its run.log with DONE.
set -u
QN=${1:?queue name}; LIST=${2:?cells list}
source /raid/work/run/gt_card_env.sh
Q=/raid/work/logs/$QN-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$Q"; ln -sfn "$Q" "/raid/work/logs/$QN-latest"
export W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-gracetier
CELLS=/raid/work/run/cells
VERIFY=/raid/work/src/garner/deepseek-v4.1-flash/verify
RESDIR=$GT/home/state/prof/gracetier-res
POL=$GT/policy.env; POLBASE=$GT/policy.env.base
REF=${IDENT_REF:-/raid/work/ident/gt275-run1.json}
[ -f "$POLBASE" ] || cp "$POL" "$POLBASE"
# The hot map is part of the tree. Every cell starts from the saved base map; a cell's pre-hook may replace it.
RMAP=$GT/tree/fastpath/hook/rowmap-allres.json; RMAPBASE=$GT/rowmap-allres.base.json
[ -f "$RMAPBASE" ] || cp "$RMAP" "$RMAPBASE"

down() { ( cd "$GT" && WORK="$GT" DS41_HOME="$GT/home" bash "$GT/launch-gt1.sh" down ) > "$Q/$1-down.log" 2>&1; docker rm -f ds41 ds41-sidecar >/dev/null 2>&1; sleep 8; }
up() {
  cp "$POLBASE" "$POL"
  cp "$RMAPBASE" "$RMAP"
  [ -f "$CELLS/$1.env" ] && cat "$CELLS/$1.env" >> "$POL"
  echo "policy additions: $( [ -f "$CELLS/$1.env" ] && tr '\n' ' ' < "$CELLS/$1.env" | cut -c1-400 || echo none)"
  if [ -f "$CELLS/$1.pre.sh" ]; then bash "$CELLS/$1.pre.sh" 2>&1 | tail -3 | cut -c1-250; fi
  ( cd "$GT" && WORK="$GT" MODEL_DIR="$MODEL_DIR" DS41_HOME="$GT/home" bash "$GT/launch-gt1.sh" up ) > "$Q/$1-up.log" 2>&1
  local rc=$?
  if ! curl -sf -m 5 "$W0_BASE/v1/models" | grep -q "$W0_MODEL"; then
    echo "NOT SERVING (up rc=$rc)"
    L=$(ls -t "$GT"/home/state/logs/*-ds41.log 2>/dev/null | head -1)
    [ -n "$L" ] && tr -d '\r' < "$L" | grep -E "RuntimeError|Error:|refus|FATAL" | grep -v "Engine core initialization failed" | sed -E 's/^\(EngineCore[^]]*\] //' | tail -3 | cut -c1-400
    return 1
  fi
  echo "serving at $(date -u +%T); $(docker logs ds41 2>&1 | grep -m1 'non-default args' | grep -oE "num_speculative_tokens_per_batch_size': \[\[[^']*\]\]")"
  docker logs ds41 2>&1 | grep -E "ENGAGED|GPU KV cache size|K1 override|captured FULL 1-request|dynamic K: added" | sed -E 's/; VMM.*//; s/^\(EngineCore[^]]*\] //' | cut -c1-230
  bash /raid/work/run/smoke.sh 30006 "$W0_MODEL" 2>&1 | tail -1
}
pass() { ( cd "$DEC" && timeout 1800 python3 bench/gt_mix_drive.py pass --set "$2" --out "$Q/$1.json" > "$Q/$1.log" 2>&1 ); echo "$1: $(tail -1 "$Q/$1.log" | sed -E 's/; per-prompt.*//; s/^pass [AB]: 20\/20 prompts, //' | cut -c1-170)"; }
bf() {  # label, skip list, extra args...
  local label=$1 skip=$2; shift 2
  ( cd "$VERIFY" && python3 -I bench_full.py --base-url "$W0_BASE" --key-file /raid/work/run/no-key --out "$Q/$label.json" --skip "$skip" "$@" ) > "$Q/$label.log" 2>&1
  local rc=$?
  grep -E '^\| ' "$Q/$label.md" 2>/dev/null | grep -E '[0-9]' | grep -vE '^\| (shape|streams|prompt tokens) ' | sed -E "s/^/$label: /" | cut -c1-220
  grep -E 'preemptions|^\| tool call' "$Q/$label.md" 2>/dev/null | sed -E "s/^/$label: /" | cut -c1-330
  [ $rc -eq 0 ] || { echo "$label: bench_full rc=$rc"; tail -2 "$Q/$label.log" | cut -c1-200; }
}
ident() {  # label, concurrency
  ( cd /raid/work/run && python3 ident_capture.py "$W0_BASE" "$W0_MODEL" "/raid/work/ident/$QN-$1.json" A,B 256 "$2" 2>&1 | tail -1 | cut -c1-200
    python3 ident_cmp.py "$REF" "/raid/work/ident/$QN-$1.json" | sed -n 2p | sed -E "s/^/$1 vs reference: /" | cut -c1-200
    python3 ident_diag.py "$REF" "/raid/work/ident/$QN-$1.json" | sed -n '2,5p' | cut -c1-260 )
}
ALL=fixed,gsm8k_c1,greedy_c1,coding,streams,prefill,capacity,quality
only() { echo "$ALL" | tr ',' '\n' | grep -vx "$1" | paste -sd, -; }
# LSOAK-BEGIN
lsoak() {  # cell name, rounds. Long soak: each round is five 10-minute windows (2, 4, 8, 16, 32 sampled streams) and
           # sets A and B, while uncached long prompts (14K / 57K / 114K) arrive every half minute for the whole run
           # and GPU and host memory are sampled every minute. One round is about 53 minutes.
  local name=$1 R=$2 r
  rm -f "$Q/$name-lsoak.stop" "$Q/$name-lsoak-mem.csv" "$Q/$name-lsoak.pfcount"
  ( i=0; while [ ! -f "$Q/$name-lsoak.stop" ]; do i=$((i + 1)); bf "$name-lsoakpf$i" "$(only prefill)" --prefill-reps 1 > /dev/null 2>&1 < /dev/null; echo "$i" > "$Q/$name-lsoak.pfcount"; sleep 30; done ) &
  ( while [ ! -f "$Q/$name-lsoak.stop" ]; do echo "$(date -u +%T),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1),$(free -m | awk '/^Mem:/{print $3","$7}'),$(docker inspect -f '{{.RestartCount}}' ds41 2>/dev/null)" >> "$Q/$name-lsoak-mem.csv"; sleep 60; done ) &
  ( sleep 150; echo "$name-lsoak: after 150 s: prefill rounds $(cat "$Q/$name-lsoak.pfcount" 2>/dev/null) with $(ls "$Q"/"$name"-lsoakpf*.json 2>/dev/null | wc -l | tr -d ' ') results; memory samples $(wc -l < "$Q/$name-lsoak-mem.csv" 2>/dev/null | tr -d ' ')" ) &
  for r in $(seq 1 "$R"); do
    bf "$name-lsoak-r$r" "$(only streams)" --conc 2,4,8,16,32 --window 600 --warmup 20 < /dev/null
    pass "$name-lsoak-r$r-a" A; pass "$name-lsoak-r$r-b" B
    echo "$name-lsoak: round $r of $R done $(date -u +%T); prefill rounds $(cat "$Q/$name-lsoak.pfcount" 2>/dev/null) with $(ls "$Q"/"$name"-lsoakpf*.json 2>/dev/null | wc -l | tr -d ' ') results; server log error lines $(docker logs ds41 2>&1 | grep -cE ' ERROR |Traceback'); restarts $(docker inspect -f '{{.RestartCount}}' ds41 2>/dev/null)"
  done
  touch "$Q/$name-lsoak.stop"; wait
  awk -F, -v n="$name" '{g=$2+0; h=$3+0; a=$4+0; if(NR==1){g0=g;h0=h;a0=a;gmin=g;gmax=g;hmin=h;hmax=h;amin=a} if(g<gmin)gmin=g; if(g>gmax)gmax=g; if(h<hmin)hmin=h; if(h>hmax)hmax=h; if(a<amin)amin=a; g1=g;h1=h;a1=a} END{printf "%s-lsoak memory (%d samples): GPU MiB first %d last %d min %d max %d; host used MiB first %d last %d min %d max %d; host available MiB first %d last %d min %d\n", n, NR, g0, g1, gmin, gmax, h0, h1, hmin, hmax, a0, a1, amin}' "$Q/$name-lsoak-mem.csv"
}
# LSOAK-END
# HO-BEGIN
tele_n() { if [ -f "$RESDIR/telemetry.jsonl" ]; then wc -l < "$RESDIR/telemetry.jsonl" | tr -d ' '; else echo 0; fi; }
tele_since() {  # label, number of telemetry lines to skip: share of expert picks that were cold since then
  python3 - "$RESDIR/telemetry.jsonl" "$2" "$1" <<'PY'
import json, sys
path, n0, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cold = sel = k = 0
moved = None
try:
    lines = open(path).read().splitlines()[n0:]
except OSError:
    lines = []
for line in lines:
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("event") != "drain":
        continue
    cold += d.get("cold", 0); sel += d.get("sel", 0); k += 1; moved = d.get("hot_moved", moved)
share = "%.2f %%" % (100.0 * cold / sel) if sel else "n/a"
print("%s: cold picks %d of %d = %s over %d drains; hot experts moved since boot %s" % (label, cold, sel, share, k, moved))
PY
}
heldout() {  # cell name. Fit the placement on set A only, then time set B on first contact, then settled.
  local name=$1 n0
  n0=$(tele_n); pass "$name-ho-A-first" A; tele_since "$name-ho set A, first contact on the boot map" "$n0"
  n0=$(tele_n); pass "$name-ho-A-2" A; pass "$name-ho-A-3" A; tele_since "$name-ho set A, passes 2 and 3" "$n0"
  n0=$(tele_n); pass "$name-ho-B-first" B; tele_since "$name-ho set B, first contact (held out)" "$n0"
  n0=$(tele_n); pass "$name-ho-B-2" B; pass "$name-ho-B-3" B; tele_since "$name-ho set B, passes 2 and 3" "$n0"
  n0=$(tele_n); pass "$name-ho-A-again" A; tele_since "$name-ho set A again" "$n0"
  n0=$(tele_n); bf "$name-ho-fixed" "$(only fixed)" --reps 5 < /dev/null; tele_since "$name-ho code prompt, never seen in this boot" "$n0"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2] + ' code prompt runs, tok/s: ' + ', '.join(str(r.get('out_tok_s')) for r in d['fixed']['runs']))" "$Q/$name-ho-fixed.json" "$name-ho" 2>/dev/null
}
# HO-END

{
  date -u +'start %FT%TZ'
  while IFS='|' read -r name profile <&3; do
    [ -n "$name" ] || continue
    echo "== CELL $name  [$profile]  $(date -u +%T)"
    down "$name" < /dev/null
    if ! up "$name" < /dev/null; then echo "CELL_DONE $name FAILED_BOOT"; continue; fi
    [ -f "$CELLS/$name.nowarm" ] || for s in A B A B; do pass "$name-warm-$s" "$s" > /dev/null; done
    idx=0
    for tok in $profile; do
      idx=$((idx + 1))
      case "$tok" in
        c1) for r in 1 2; do pass "$name-c1-r$r-a" A; pass "$name-c1-r$r-b" B; done
            bf "$name-fixed" "$(only fixed)" --reps 6 < /dev/null ;;
        q) bf "$name-quality" "$(only quality)" < /dev/null ;;
        full) bf "$name-score" "" --conc 2,4,8,16,32 --reps 5 < /dev/null ;;          # every stage of the public scoreboard
        fullnc) bf "$name-score" "capacity" --conc 2,4,8,16,32 --reps 5 < /dev/null ;;  # all but the capacity stage
        a5x3) for r in 1 2 3; do   # Meadlock's A5 cells (the ruler behind his published one-GPU numbers), three passes
                ( cd "$DEC" && A5_OUT="$Q/$name-a5r$r" A5_ARM="$name-r$r" A5_WARM=0 A5_GAUNTLET=0 bash bench/box/a5-cells.sh 2>&1 \
                  | grep -vE "cells done" | sed -E 's/^[0-9-]+ [0-9:]+ [A-Z]+ //; s/; per-prompt.*//; s/; per-stream.*//' | cut -c1-200 ) < /dev/null
              done ;;
        soak) # 10 minutes at 16 sampled streams while uncached long prompts (14K / 57K / 114K) arrive every half minute
              ( for i in 1 2 3 4 5 6 7 8; do bf "$name-soakpf$i" "$(only prefill)" --prefill-reps 1 > /dev/null 2>&1 < /dev/null; sleep 30; done ) &
              bf "$name-soak16" "$(only streams)" --conc 16 --window 600 --warmup 20 < /dev/null
              wait
              echo "$name-soak: prefill rounds done $(ls "$Q"/$name-soakpf*.json 2>/dev/null | wc -l); server log error lines $(docker logs ds41 2>&1 | grep -cE ' ERROR |Traceback'); restarts $(docker inspect -f '{{.RestartCount}}' ds41 2>/dev/null); gpu MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" ;;
        lsoak:*) lsoak "$name" "${tok#lsoak:}" ;;   # long soak, N rounds of about 53 minutes (see lsoak above)
        ho) heldout "$name" ;;                       # held-out placement test (needs cells/<name>.nowarm to skip the warm-up)
        s:*) spec=${tok#s:}; list=${spec%x*}; n=${spec##*x}; [ "$n" = "$spec" ] && n=1
             for r in $(seq 1 "$n"); do bf "$name-streams$idx-r$r" "$(only streams)" --conc "$list" < /dev/null; done ;;
        pf) bf "$name-prefill" "$(only prefill)" < /dev/null ;;
        id) ident "$name-c1" 1 < /dev/null ;;
        idc:*) ident "$name-c${tok#idc:}" "${tok#idc:}" < /dev/null ;;
        res:*) f=$CELLS/${tok#res:}
               doc=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('seq',None); print(json.dumps(d))" "$f")
               echo "residency control: $doc"
               ( cd "$DEC" && timeout 400 python3 bench/gt_res_drive.py ctrl --dir "$RESDIR" --set "$doc" --nudge step --wait ack --timeout 300 2>&1 | tail -1 | cut -c1-300 ) < /dev/null ;;
        *) echo "unknown profile token $tok" ;;
      esac
    done
    if [ -f "$CELLS/$name.post.sh" ]; then bash "$CELLS/$name.post.sh" "$Q" < /dev/null 2>&1 | tail -n 100 | cut -c1-230; fi
    echo "after: $(docker ps --format '{{.Names}} {{.Status}}'), restarts $(docker inspect -f '{{.RestartCount}}' ds41 2>/dev/null), telemetry: $(tail -n 1 "$RESDIR/telemetry.jsonl" 2>/dev/null | cut -c1-260)"
    echo "CELL_DONE $name $(date -u +%T)"
    if [ "$name" = ctl ] && ! grep -q "ctl-c1 vs reference: identical outputs: 40/40" "$Q/run.log"; then
      echo "ABORT: the control cell does not reproduce the reference output, so the patches are not inert"; break
    fi
  done 3< <(grep -v '^#' "$LIST")
  cp "$POLBASE" "$POL"
  cp "$RMAPBASE" "$RMAP"
  date -u +'end %FT%TZ'
  echo DONE
} > "$Q/run.log" 2>&1
