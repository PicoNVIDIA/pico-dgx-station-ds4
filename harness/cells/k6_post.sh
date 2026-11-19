#!/bin/bash
# Print what the K6 counters say for the boot that just ran.
docker logs ds41 2>&1 | grep -E "K6 (stage skip|scratch lanes)" | grep -vE "steps (64|128|192|256) " | tail -n 5 | sed -E 's/^.*DS41_FASTPATH gracetier: //' | cut -c1-300
docker logs ds41 2>&1 | grep -cE "K6 stage skip ERROR" | sed 's/^/K6 error lines: /'
grep '"stage_skip"' /raid/work/gt/gracetier/home/state/prof/gracetier-res/telemetry.jsonl 2>/dev/null | tail -n 1 | cut -c1-330
