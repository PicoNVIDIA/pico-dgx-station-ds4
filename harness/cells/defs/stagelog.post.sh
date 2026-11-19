#!/bin/bash
set -u
Q=$1; GT=/raid/work/gt/gracetier; P=/raid/work/run/patches/K4
L=$GT/home/state/prof/gracetier-res/k4-stage.jsonl
docker logs ds41 2>&1 | grep -E "K4 stage log" | tail -2 | cut -c1-220
( cd "$GT" && WORK="$GT" DS41_HOME="$GT/home" bash "$GT/launch-gt1.sh" down ) > /dev/null 2>&1; docker rm -f ds41 > /dev/null 2>&1
cp "$L" "$Q/k4-stage.jsonl" 2>/dev/null || sudo -n cp "$L" "$Q/k4-stage.jsonl"; sudo -n chown "$(id -u):$(id -g)" "$Q/k4-stage.jsonl" 2>/dev/null
echo "stage log: $(wc -l < "$Q/k4-stage.jsonl") lines, $(du -h "$Q/k4-stage.jsonl" | cut -f1)"
python3 $P/k4_stage_analyze.py "$Q/k4-stage.jsonl" --max-ntok 300 --json "$Q/k4-analysis.json" > "$Q/k4-analysis.txt" 2>&1; echo "analyze rc=$? -> $Q/k4-analysis.txt ($(wc -l < "$Q/k4-analysis.txt") lines)"
head -n 70 "$Q/k4-analysis.txt"
cd "$GT" && patch -p1 -R < $P/0001-gracetier-stage-log.patch | tail -1
python3 - <<'PY'
p = "/raid/work/gt/gracetier/tree/serve/serve.sh"
s = open(p).read()
keep = [l for l in s.splitlines(keepends=True) if "GRACETIER_STAGE_LOG=" not in l]
open(p, "w").write("".join(keep)); print("serve.sh: stage-log env line removed")
PY
