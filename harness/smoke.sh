#!/bin/bash
# One real request against a local OpenAI-compatible server. usage: smoke.sh <port> <model>
P=${1:?port}; M=${2:?model}
curl -s -m 120 http://127.0.0.1:$P/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Reply with just the number.\"}],\"max_tokens\":200,\"temperature\":0}" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
m = d['choices'][0]['message']
c = (m.get('content') or '').strip()
r = (m.get('reasoning_content') or m.get('reasoning') or '')
print('content:', repr(c[:80]), '| reasoning chars:', len(r), '| usage:', d.get('usage'))
print('SMOKE', 'PASS' if '391' in c else 'FAIL')
"
