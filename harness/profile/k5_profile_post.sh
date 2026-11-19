#!/bin/bash
# After a boot under the nsys wrap: warm, profile one range, wait for the server to exit, export, print the table.
# usage: k5_profile_post.sh TAG code|general TOKENS
set -u
TAG=$1; MODE=$2; TOK=${3:-512}
GT=/raid/work/gt/gracetier; K5=/raid/work/run/patches/K5; OUT=/raid/work/logs/k5/$TAG; mkdir -p "$OUT"
docker inspect -f '{{json .Args}}' ds41 > "$OUT/ds41-args.json" 2>/dev/null
grep -q '/nsys' "$OUT/ds41-args.json" && echo "ds41 runs under the nsys wrap" || { echo "ds41 has NO nsys wrap: no profile taken"; exit 1; }
PARGS=()
if [ "$MODE" = code ]; then
  python3 -c "
import json, secrets, sys
t = open('/raid/work/src/garner/deepseek-v4.1-flash/verify/data/fixed_prompt.txt').read()
open(sys.argv[1], 'w').write(json.dumps({'prompt': '[cache-bust nonce: ' + secrets.token_hex(16) + ']\n' + t}) + '\n')" "$OUT/prompts.jsonl"
  PARGS=(--prompts "$OUT/prompts.jsonl")
fi
python3 "$K5/k5_drive.py" "$OUT/warm.json" --streams 1 --max-tokens "$TOK" --warm 3 "${PARGS[@]}" 2>&1 | tail -n 1 | cut -c1-300
timeout 900 python3 "$K5/k5_drive.py" "$OUT/cell.json" --streams 1 --max-tokens "$TOK" --warm 0 --profile "${PARGS[@]}" 2>&1 | tail -n 1 | cut -c1-400
for i in $(seq 1 90); do docker ps --format '{{.Names}}' | grep -qx ds41 || break; sleep 5; done
docker ps --format '{{.Names}}' | grep -qx ds41 && echo "server still up after the profile (unexpected)"
docker logs ds41 2>&1 | grep -iE "nsys|Generating|report file|Collecting" | tail -n 4 | cut -c1-200
PROF=$GT/home/state/prof/$TAG
rep=$(sudo -n ls -t "$PROF" 2>/dev/null | grep '\.nsys-rep$' | head -1)
[ -n "$rep" ] || { echo "no .nsys-rep under $PROF:"; sudo -n ls -la "$PROF" 2>&1 | head -5; exit 1; }
sudo -n /usr/local/bin/nsys export --type sqlite --force-overwrite=true -o "$OUT/$TAG.sqlite" "$PROF/$rep" > "$OUT/export.log" 2>&1; echo "nsys export rc=$? ($(du -h "$OUT/$TAG.sqlite" 2>/dev/null | cut -f1))"
sudo -n chown -R "$(id -u):$(id -g)" "$OUT"
python3 "$K5/k5_step_table.py" "$OUT/$TAG.sqlite" --top 70 --csv "$OUT/table.csv" --json "$OUT/table.json" > "$OUT/table.txt" 2>&1; echo "table rc=$? -> $OUT/table.txt ($(wc -l < "$OUT/table.txt") lines)"
head -n 80 "$OUT/table.txt"
