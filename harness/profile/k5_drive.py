#!/usr/bin/env python3
"""k5_drive.py: drive one profiled cell on the gracetier server. Stdlib only.
Modelled on Meadlock's w0/w0_drive.py 'step --profile' (gracetier-fadc3d2/w0/w0_drive.py:127-165), plus --streams.

  python3 k5_drive.py OUT.json [--streams N] [--max-tokens 256] [--warm 2] [--profile] [--prompts FILE.jsonl]

Warm (unprofiled) generations first, so graphs, prefix cache and the residency map are settled.
With --profile: POST /start_profile, run N concurrent greedy streams, POST /stop_profile.
An nsys-wrapped server (capture-range-end=stop-shutdown) exits after /stop_profile. That is expected.
Reads /metrics before and after: steps = vllm:spec_decode_num_drafts delta, tokens = generation_tokens delta.
Env: K5_BASE (default http://127.0.0.1:30006), K5_MODEL (default dsv41-flash-gracetier).
--prompts: jsonl with a "prompt" (or "text") field per line; default is Meadlock's STEP_PROMPT.
"""
import json, os, sys, threading, time, urllib.request, urllib.error

BASE = os.environ.get("K5_BASE", "http://127.0.0.1:30006")
MODEL = os.environ.get("K5_MODEL", "dsv41-flash-gracetier")
STEP_PROMPT = ("Write a detailed technical explanation, about 600 words, of how speculative decoding with a draft head "
               "works in a mixture-of-experts transformer, including why each verify step touches more unique experts "
               "than a plain decode step. Then write a Python function that simulates acceptance length for a given "
               "per-position acceptance probability vector.")


def post_raw(path, timeout=60):
    req = urllib.request.Request(BASE + path, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()[:200].decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:200].decode(errors="replace")
    except Exception as e:  # the server exiting after stop_profile is expected
        return -1, repr(e)[:200]


def metrics():
    want = {"vllm:spec_decode_num_drafts_total": "drafts", "vllm:spec_decode_num_accepted_tokens_total": "accepted",
            "vllm:generation_tokens_total": "gen"}
    out = {}
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            for w, k in want.items():
                if line.startswith(w + "{"):
                    out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def stream(prompt, max_tokens, res, i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1,
            "max_tokens": max_tokens, "seed": 1, "chat_template_kwargs": {"thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = tl = None; usage = None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                p = line[5:].strip()
                if p == b"[DONE]":
                    break
                d = json.loads(p)
                if d.get("usage"):
                    usage = d["usage"]
                ch = (d.get("choices") or [{}])[0]
                if ch.get("delta", {}).get("content"):
                    tl = time.time(); tf = tf or tl
        res[i] = {"ttft_s": round((tf or time.time()) - t0, 3), "decode_s": round(tl - tf, 3) if tf and tl else None,
                  "completion_tokens": (usage or {}).get("completion_tokens")}
    except Exception as e:
        res[i] = {"error": repr(e)[:200]}


def run(prompts, n, max_tokens):
    res = [None] * n
    th = [threading.Thread(target=stream, args=(prompts[i % len(prompts)], max_tokens, res, i)) for i in range(n)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    return res, time.time() - t0


def main(a):
    if len(a) < 2:
        print(__doc__); return 2
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    n = int(opt("--streams", "1")); mt = int(opt("--max-tokens", "256")); warm = int(opt("--warm", "2"))
    prompts = [STEP_PROMPT]
    if "--prompts" in a:
        prompts = [(json.loads(l).get("prompt") or json.loads(l).get("text")) for l in open(opt("--prompts", "")) if l.strip()]
    # salt concurrent streams so they do not collapse onto one prefix-cache entry
    if n > 1 and len(prompts) < n:
        prompts = [f"[session {i}] " + prompts[i % len(prompts)] for i in range(n)]
    out = {"base": BASE, "model": MODEL, "streams": n, "max_tokens": mt, "profile": "--profile" in a}
    for _ in range(warm):
        run(prompts, n, mt)
    m0 = metrics()
    if out["profile"]:
        out["start_profile"] = post_raw("/start_profile")
    res, wall = run(prompts, n, mt)
    if out["profile"]:
        out["stop_profile"] = post_raw("/stop_profile", timeout=600)
    try:
        m1 = metrics()
    except Exception as e:
        m1 = {"error": repr(e)[:120]}
    out.update({"wall_s": round(wall, 3), "streams_detail": res})
    if "drafts" in m0 and "drafts" in m1:
        steps = m1["drafts"] - m0["drafts"]; gen = m1["gen"] - m0["gen"]
        out.update({"steps": steps, "gen_tokens": gen, "tokens_per_step": round(gen / steps, 3) if steps else None,
                    "ms_per_step_wall": round(1000 * wall / steps, 3) if steps else None,
                    "tok_s_aggregate": round(gen / wall, 1)})
    else:
        out["metrics_after"] = m1  # server already gone: take steps from the trace instead
    json.dump(out, open(a[1], "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "streams_detail"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
