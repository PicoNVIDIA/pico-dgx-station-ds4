#!/usr/bin/env python3
"""Print the numbers of a bench_full / bench_sgl result JSON on a few lines. usage: bench_sum.py <label> <result.json>"""
import json
import sys

label, path = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as exc:
    print(f"{label}: no result ({type(exc).__name__}: {exc})"); sys.exit(0)

def med(s, k, nd=1):
    v = (s or {}).get(k) or {}
    return "n/a" if v.get("median") is None else f"{v['median']:.{nd}f} ({v.get('q1', 0):.{nd}f}-{v.get('q3', 0):.{nd}f}, n={v.get('n')})"

for st in ("fixed", "coding"):
    r = d.get(st)
    if r is None:
        continue
    if "error" in r:
        print(f"{label}: {st} ERROR {r['error'][:200]}"); continue
    s = r["summary"]
    print(f"{label}: {st}: {med(s, 'out_tok_s')} tok/s; TTFT {med(s, 'ttft_ms', 0)} ms; out tokens {med(s, 'completion_tokens', 0)}; prompt {med(s, 'prompt_tokens', 0)}")
for st in ("gsm8k_c1", "greedy_c1"):
    r = d.get(st)
    if r is None:
        continue
    print(f"{label}: {st}: " + (f"ERROR {r['error'][:200]}" if "error" in r else f"{r['aggregate_tok_s']} tok/s over {r['window_s']} s, request errors {r['request_errors']}" + (f" (server counter {r['server_counter_tok_s']})" if 'server_counter_tok_s' in r else "")))
r = d.get("streams")
if isinstance(r, dict) and "error" in r:
    print(f"{label}: streams ERROR {r['error'][:200]}")
for w in r if isinstance(r, list) else []:
    print(f"{label}: {w['conc']} streams: {w['aggregate_tok_s']} tok/s aggregate, {w['per_stream_tok_s']} a stream, request errors {w['request_errors']}" + (f" (server counter {w['server_counter_tok_s']})" if 'server_counter_tok_s' in w else ""))
r = d.get("prefill")
if r is not None:
    if "error" in r:
        print(f"{label}: prefill ERROR {r['error'][:200]}")
    for row in r.get("rows", []):
        print(f"{label}: prefill {med(row, 'prompt_tokens', 0)} tokens: {med(row, 'prefill_tok_s', 0)} tok/s, TTFT {med(row, 'ttft_ms', 0)} ms")
q = d.get("quality")
if q is not None:
    if "error" in q:
        print(f"{label}: quality ERROR {q['error'][:200]}")
    else:
        g = q.get("gsm8k_200") or {}
        n = q.get("needles_200k") or {}
        print(f"{label}: quality: tool call {(q.get('tool_call') or {}).get('passed')} [{((q.get('tool_call') or {}).get('output') or '')[:80]}]; "
              f"needles {n.get('found')}/{n.get('total')} passed {n.get('passed')}; gsm8k {json.dumps({k: v for k, v in g.items() if k != 'raw'})[:200]}{(' raw ' + str(g.get('raw'))[:120]) if g.get('raw') else ''}")
print(f"{label}: status {d.get('status')} failed stages {d.get('failed_stages')}")
