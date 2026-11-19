#!/bin/bash
# K7: forward the four packed-scale variables from policy.env into the container. ONE anchored line in
# tree/serve/serve.sh, inside start_vllm(), right after the PEER3_SPINS line (the K1 loop and the K6 lines sit at the
# same place; this insert does not depend on them). Idempotent: a second run changes nothing.
#
#   bash k7_serve_insert.sh /raid/work/gt/gracetier/tree/serve/serve.sh        # apply
#   bash k7_serve_insert.sh --remove <serve.sh>                                # take the line out again
#
# policy.env lines (NAME=value, no quotes, no comments):
#   DS41_GT_SF_PACK=a2b6          a2b6 | a2b4 | kernel-b6 | kernel-b4 | probe   -> GRACETIER_SF_PACK
#   DS41_GT_SF_PACK_ORDER=lo      lo | hi (first code of a 16-code group in the low / high bits)
#   DS41_GT_SF_PACK_AUDIT=0,19,39 layers that keep a dense bank and run MegaMoE both ways (no speed from that boot)
#   DS41_GT_SF_PACK_CP=decomp     decomp | plain (the kernel's copy to tensor memory)
# a2b6 / a2b4 also need DS41_GT_STAGE_SKIP=1 (the K6 lines) and patches 0002 + 0003.
set -euo pipefail
MARK='# K7 packed scales'
LINE='  local k7v; for k7v in DS41_GT_SF_PACK DS41_GT_SF_PACK_ORDER DS41_GT_SF_PACK_AUDIT DS41_GT_SF_PACK_CP; do if [ -n "${!k7v:-}" ]; then envs+=(-e "GRACETIER_${k7v#DS41_GT_}=${!k7v}"); fi; done  '"$MARK"
ANCHOR='envs+=(-e "PEER3_SPINS=$DS41_PEER3_SPINS")'
remove=0
if [ "${1:-}" = "--remove" ]; then remove=1; shift; fi
f=${1:?usage: k7_serve_insert.sh [--remove] <serve.sh>}
[ -f "$f" ] || { echo "k7_serve_insert: $f not found" >&2; exit 2; }
have=$(grep -cF "$MARK" "$f" || true)
if [ "$remove" = 1 ]; then
  [ "$have" = 0 ] && { echo "k7_serve_insert: no K7 line in $f"; exit 0; }
  grep -vF "$MARK" "$f" > "$f.k7tmp" && cat "$f.k7tmp" > "$f" && rm -f "$f.k7tmp"
  echo "k7_serve_insert: removed $have K7 line(s) from $f"; exit 0
fi
if [ "$have" != 0 ]; then echo "k7_serve_insert: $f already has the K7 line ($have)"; exit 0; fi
n=$(grep -cF "$ANCHOR" "$f" || true)
[ "$n" = 1 ] || { echo "k7_serve_insert: the anchor line occurs $n times in $f (expected 1); nothing changed" >&2; exit 3; }
awk -v anchor="$ANCHOR" -v line="$LINE" '{ print } index($0, anchor) { print line }' "$f" > "$f.k7tmp"
cat "$f.k7tmp" > "$f" && rm -f "$f.k7tmp"
bash -n "$f" || { echo "k7_serve_insert: $f does not parse after the insert" >&2; exit 4; }
echo "k7_serve_insert: inserted after line $(grep -nF "$ANCHOR" "$f" | cut -d: -f1) of $f"
