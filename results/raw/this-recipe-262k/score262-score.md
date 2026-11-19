# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T01:53:37+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 545.6 (535.3-588.7) | 119 (117-119) | 5.39 (5.23-5.71) | 103.2 (101.5-105.5) | 5 |
| coding prompt (2,048 out, greedy, thinking off) | 629.7 (590.3-636.9) | 159 (154-163) | 6.34 (6.30-6.35) | 99.1 (93.7-101.5) | 5 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 423.3 | — | 3.831 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 518.2 | — | 4.748 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 602 | 301.2 | 3.589 | 60 |
| 4 | 876 | 219.0 | 3.685 | 60 |
| 8 | 1,234 | 154.2 | 3.088 | 60 |
| 16 | 1,740 | 108.7 | 3.064 | 60 |
| 32 | 2,299 | 71.9 | 3.035 | 60 |

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14264 (14243-14266) | 271 (270-323) | 52422 (45284-52694) |
| 56905 (56841-56924) | 1051 (1034-1083) | 54006 (52517-54986) |
| 113720 (113688-113736) | 2139 (2123-2146) | 53136 (52990-53550) |

## Quality (same session)

| check | result |
|---|---|
| GSM8K-200 (thinking on, T=0.3) | 195/200 (97.5 %) |
| needles at 200K tokens | 3/3 |
| tool call | PASS |
