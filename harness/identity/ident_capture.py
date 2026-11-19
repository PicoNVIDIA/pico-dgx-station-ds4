#!/usr/bin/env python3
"""Capture greedy outputs for the identity check. Same request shape as Meadlock's A5 ruler
(temperature 0, top_p 1, seed 1, thinking off), non-streamed, with per-token logprobs (top 2).
usage: ident_capture.py <base_url> <model> <out.json> [sets=A,B] [max_tokens=256] [concurrency=1]
With concurrency > 1 the same prompts are sent that many at a time, so the engine batches them: comparing that
capture with a one-at-a-time capture shows whether batching alone changes the engine's output.
stdlib only. Prompts come from gb300-dsv41-decode/bench/gt_mix_drive.py (SETS)."""
import json, sys, time, urllib.request, hashlib
from concurrent.futures import ThreadPoolExecutor
DEC = "/raid/work/src/gb300-dsv41-decode"
sys.path[:0] = [DEC + "/bench", DEC + "/w0", DEC]
import gt_mix_drive as G

base, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
sets = (sys.argv[4] if len(sys.argv) > 4 else "A,B").split(",")
max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 256
conc = int(sys.argv[6]) if len(sys.argv) > 6 else 1

def ask(prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1,
            "max_tokens": max_tokens, "seed": 1, "chat_template_kwargs": {"thinking": False}, "stream": False,
            "logprobs": True, "top_logprobs": 2}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

def one(task):
    s, i, kind, prompt = task
    rec = {"set": s, "i": i, "kind": kind, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16]}
    try:
        d = ask(prompt)
        ch = d["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        rec["text"] = ch["message"].get("content") or ""
        rec["finish"] = ch.get("finish_reason")
        rec["tokens"] = [t["token"] for t in lp]
        rec["lp1"] = [round(t["logprob"], 5) for t in lp]
        rec["lp2"] = [round(t["top_logprobs"][1]["logprob"], 5) if len(t.get("top_logprobs") or []) > 1 else None for t in lp]
        rec["tok2"] = [t["top_logprobs"][1]["token"] if len(t.get("top_logprobs") or []) > 1 else None for t in lp]
        rec["usage"] = d.get("usage")
    except Exception as e:
        rec["error"] = repr(e)[:300]
    return rec

tasks = [(s, i, kind, prompt) for s in sets for i, (kind, prompt) in enumerate(G.SETS[s])]
t0 = time.time()
if conc > 1:
    with ThreadPoolExecutor(max_workers=conc) as ex:
        rows = list(ex.map(one, tasks))
else:
    rows = [one(t) for t in tasks]
json.dump({"base": base, "model": model, "sets": sets, "max_tokens": max_tokens, "concurrency": conc,
           "t": time.time(), "rows": rows}, open(out, "w"))
ok = [r for r in rows if "error" not in r]
print(f"captured {len(ok)}/{len(rows)} prompts, {sum(len(r['tokens']) for r in ok)} tokens with logprobs, "
      f"concurrency {conc}, {time.time()-t0:.0f}s -> {out}")
