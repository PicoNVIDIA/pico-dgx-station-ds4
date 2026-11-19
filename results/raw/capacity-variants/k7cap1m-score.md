# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T10:03:03+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 507.0 (502.1-539.0) | 132 (128-138) | 5.30 (5.30-5.44) | 97.3 (94.9-99.0) | 5 |
| coding prompt (2,048 out, greedy, thinking off) | 580.4 (500.1-586.3) | 174 (169-179) | 6.18 (6.13-6.34) | 95.0 (88.2-95.0) | 5 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 399.4 | — | 3.813 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 487.3 | — | 4.726 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 576 | 288.2 | 3.677 | 60 |
| 4 | 807 | 201.7 | 3.611 | 60 |
| 8 | 1,143 | 142.9 | 3.032 | 60 |
| 16 | 1,574 | 98.4 | 3.026 | 60 |
| 32 | 2,084 | 65.1 | 3.057 | 60 |

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14275 (14240-14280) | 288 (287-339) | 49491 (43069-49563) |
| 56835 (56824-56849) | 1077 (1074-1103) | 52791 (51585-52932) |
| 113751 (113665-113782) | 2248 (2246-2250) | 50550 (50539-50615) |

## Long-context capacity

GPU KV cache: ? tokens. 16 distinct ~110,000-token contexts decoding 2,500 forced tokens each, plus one needle request: not all ran at once (max running 16.0, max waiting 13.0), 0 preemptions, peak KV use 52 %, wall 65.0 s, all requests correct.

## Quality (same session)

| check | result |
|---|---|
| GSM8K-200 (thinking on, T=0.3) | 197/200 (98.5 %) |
| needles at 200K tokens | 3/3 |
| tool call | PASS |
