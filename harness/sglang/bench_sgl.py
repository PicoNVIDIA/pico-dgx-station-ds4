#!/usr/bin/env python3
"""Garner's bench_full.py against an SGLang server. Same prompts, same stages, same arguments.

bench_full.py reads vLLM's /metrics (names vllm:*) and vLLM's streaming usage. SGLang differs in three ways, and this
wrapper only bridges those; nothing about the workload changes:
  * /metrics exists only with --enable-metrics and its counters are named sglang:*  -> renamed to vllm:* here
    (the timed windows need generation_tokens; the spec-decode counters have no SGLang twin, so tokens per step
    and steps per second come out empty);
  * a streamed reply is timed from its content chunks when usage is not sent with every chunk; the token count is
    the server's final usage.completion_tokens;
  * vLLM's `cache_salt` request field is dropped (the prefill prompts are distinct random text anyway).
A stage that fails no longer aborts the run when the markdown table is written.

usage: python3 -I bench_sgl.py <the arguments of bench_full.py>
"""
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.environ.get("BENCH_FULL_DIR", "/raid/work/src/garner/deepseek-v4.1-flash/verify"))
import bench_full as B  # noqa: E402


class SglClient(B.Client):
    def counters(self) -> dict:
        try:
            text = urllib.request.urlopen(self.base + "/metrics", timeout=10).read().decode()
        except Exception:
            return {}
        out: dict = {}
        for m in re.finditer(r"^sglang:([a-z_]+?)(?:_total)?(?:\{[^}]*\})? ([0-9.e+-]+)$", text, re.M):
            k = "vllm:" + m[1]
            out[k] = out.get(k, 0.0) + float(m[2])
        return out

    def stream(self, body: dict) -> dict:
        body = dict(body, model=self.model, stream=True,
                    stream_options={"include_usage": True, "continuous_usage_stats": True})
        body.pop("cache_salt", None)
        t0 = time.perf_counter()
        first = last = None
        count = chunks = 0
        usage: dict = {}
        with self.request(body) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                ev = json.loads(line[5:])
                usage = ev.get("usage") or usage
                n = usage.get("completion_tokens", 0) if usage else 0
                has = any((ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content")
                          for ch in ev.get("choices") or [])
                chunks += 1 if has else 0
                if has or n > count:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                count = max(count, n)
        if first is None:
            raise RuntimeError("the server streamed no tokens")
        count = count or chunks
        span = last - first
        return dict(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=count,
                    ttft_ms=round((first - t0) * 1000, 1),
                    out_tok_s=round((count - 1) / span, 1) if count > 1 and span > 0 else None,
                    tokens_per_step=None, steps_s=None, content_chunks=chunks)


_markdown = B.markdown


def markdown(rec: dict) -> str:
    try:
        return _markdown(rec)
    except Exception as exc:
        return f"(no table: {type(exc).__name__}: {exc}; read the JSON)\n"


def window(c, conc: int, a, *, greedy: bool) -> dict:
    """bench_full.window with the tokens counted by the CLIENT inside the window.

    Why: SGLang adds a request's tokens to sglang:generation_tokens_total only when the request FINISHES
    (metrics_collector.observe_one_finished_request), so a counter read over a 60 s window misses every request that is
    still running when the window closes. vLLM's counter moves every step. Here each stream's usage.completion_tokens
    deltas (or content chunks, when usage comes only at the end) are time-stamped as they arrive and summed over the
    window. The server counter over the same window is kept beside it as server_counter_tok_s.
    """
    import secrets
    import threading
    questions = [json.loads(line)["question"] for line in open(B.HERE / "data/gsm8k-200.jsonl")]
    errors, buckets = [], []

    def worker(stop, seed):
        i, mine = seed, []
        buckets.append(mine)   # read by the main thread; list.append is atomic
        while not stop.is_set():
            body = {"model": c.model, "max_tokens": 4096, "stream": True,
                    "stream_options": {"include_usage": True, "continuous_usage_stats": True},
                    "messages": [{"role": "user", "content": questions[i % len(questions)]}]}
            if greedy:
                body.update(temperature=0.0, chat_template_kwargs={"thinking": False})
            else:
                body.update(temperature=1.0, top_p=0.95, chat_template_kwargs={"thinking": True})
            try:
                with c.request(body) as r:
                    counted = 0
                    for raw in r:
                        if stop.is_set():
                            break
                        line = raw.decode().strip()
                        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                            continue
                        ev = json.loads(line[5:])
                        n = (ev.get("usage") or {}).get("completion_tokens")
                        if n is not None:
                            add = max(0, n - counted)
                        else:
                            add = 1 if any((ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content")
                                           for ch in ev.get("choices") or []) else 0
                        if add:
                            counted += add
                            mine.append((time.time(), add))
            except OSError as exc:
                errors.append(repr(exc)[:200])
                time.sleep(1)
            i += 1

    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(stop, secrets.randbelow(200)), daemon=True) for _ in range(conc)]
    for t in threads:
        t.start()
    time.sleep(a.warmup)
    x, t0 = c.counters(), time.time()
    time.sleep(a.window)
    y, t1 = c.counters(), time.time()
    stop.set()
    for t in threads:
        t.join(timeout=120)
    agg = sum(n for b in buckets for ts, n in list(b) if t0 <= ts < t1) / (t1 - t0)
    d = {k: y.get(k, 0.0) - x.get(k, 0.0) for k in y}
    drafts = d.get("vllm:spec_decode_num_drafts", 0.0)
    accepted = d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    return dict(conc=conc, greedy=greedy, window_s=round(t1 - t0, 1), warmup_s=a.warmup,
                aggregate_tok_s=round(agg, 1), per_stream_tok_s=round(agg / conc, 1),
                server_counter_tok_s=round(d.get("vllm:generation_tokens", 0.0) / (t1 - t0), 1),
                tokens_per_step=round(1 + accepted / drafts, 3) if drafts else None,
                preemptions=d.get("vllm:num_preemptions", 0.0), request_errors=len(errors), counted="client")


B.Client = SglClient
B.markdown = markdown
if os.environ.get("BENCH_SGL_SERVER_COUNTER") != "1":
    B.window = window
if os.environ.get("BENCH_SGL_VLLM_NAMES") == "1":   # run against a vLLM-based server: keep its own counters
    SglClient.counters = B.Client.__mro__[1].counters
if __name__ == "__main__":
    B.main()
