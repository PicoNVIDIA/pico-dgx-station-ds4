# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T06:00:26+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 566.8 (545.6-591.8) | 122 (121-125) | 5.57 (5.47-5.63) | 102.2 (100.0-103.5) | 5 |
| coding prompt (2,048 out, greedy, thinking off) | 610.0 (565.9-623.5) | 163 (162-171) | 6.10 (5.90-6.23) | 96.6 (92.9-100.2) | 5 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 417.9 | — | 3.783 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 510.8 | — | 4.694 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 626 | 313.0 | 3.877 | 60 |
| 4 | 860 | 215.1 | 3.67 | 60 |
| 8 | 1,204 | 150.5 | 3.066 | 60 |
| 16 | 1,682 | 105.2 | 3.044 | 60 |
| 32 | 2,237 | 69.9 | 3.053 | 60 |

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14218 (14218-14234) | 310 (294-338) | 45920 (42373-48576) |
| 56873 (56806-56875) | 1067 (1064-1078) | 53182 (52684-53408) |
| 113659 (113658-113664) | 2149 (2138-2156) | 52894 (52711-53155) |

## Long-context capacity

GPU KV cache: ? tokens. 16 distinct ~110,000-token contexts decoding 2,500 forced tokens each, plus one needle request: not all ran at once (max running 11.0, max waiting 13.0), 0 preemptions, peak KV use 93 %, wall 71.1 s, all requests correct.

## Quality (same session)

| check | result |
|---|---|
| GSM8K-200 (thinking on, T=0.3) | 196/200 (98.0 %) |
| needles at 200K tokens | 3/3 |
| tool call | PASS |
