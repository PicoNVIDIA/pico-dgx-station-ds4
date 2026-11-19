# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T21:39:39+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 167.8 (166.1-168.4) | 248 (245-260) | 3.50 (3.49-3.53) | 47.7 (47.6-47.7) | 5 |
| coding prompt (2,048 out, greedy, thinking off) | 171.2 (170.7-171.7) | 338 (337-338) | 3.69 (3.67-3.70) | 46.5 (46.5-46.6) | 5 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 130.8 | — | 2.827 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 157.8 | — | 3.397 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 171 | 85.4 | 2.878 | 60 |
| 4 | 200 | 49.9 | 2.663 | 60 |
| 8 | 278 | 34.7 | 2.785 | 60 |
| 16 | 397 | 24.8 | 2.782 | 60 |
| 32 | 505 | 15.8 | 2.859 | 60 |

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14255 (14246-14258) | 527 (525-562) | 27034 (25476-27123) |
| 56837 (56816-56892) | 1987 (1985-1991) | 28606 (28574-28618) |
| 113691 (113681-113696) | 4030 (4023-4034) | 28210 (28180-28262) |

## Long-context capacity

GPU KV cache: ? tokens. 16 distinct ~110,000-token contexts decoding 2,500 forced tokens each, plus one needle request: all ran at once (max running 17.0, max waiting 15.0), 0 preemptions, peak KV use 56 %, wall 146.8 s, all requests correct.

## Quality (same session)

| check | result |
|---|---|
| GSM8K-200 (thinking on, T=0.3) | 193/200 (96.5 %) |
| needles at 200K tokens | 3/3 |
| tool call | PASS |
