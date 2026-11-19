# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T01:33:31+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 564.7 (522.4-565.4) | 126 (125-132) | 5.57 (5.18-5.79) | 101.1 (97.9-101.7) | 5 |
| coding prompt (2,048 out, greedy, thinking off) | 605.3 (583.5-611.8) | 172 (167-178) | 6.36 (6.36-6.36) | 95.2 (90.1-98.3) | 5 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 400.6 | — | 3.688 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 501.3 | — | 4.679 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 591 | 295.7 | 3.643 | 60 |
| 4 | 849 | 212.3 | 3.686 | 60 |
| 8 | 1,178 | 147.2 | 3.086 | 60 |
| 16 | 1,638 | 102.3 | 3.066 | 60 |
| 32 | 2,148 | 67.1 | 3.068 | 60 |

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14224 (14216-14262) | 284 (284-322) | 49937 (44889-50055) |
| 56788 (56786-56843) | 1101 (1082-1111) | 51660 (51159-52557) |
| 113709 (113693-113754) | 2200 (2191-2213) | 51686 (51403-51893) |

## Long-context capacity

GPU KV cache: ? tokens. 16 distinct ~110,000-token contexts decoding 2,500 forced tokens each, plus one needle request: not all ran at once (max running 15.0, max waiting 13.0), 0 preemptions, peak KV use 99 %, wall 76.1 s, all requests correct.

## Quality (same session)

| check | result |
|---|---|
| GSM8K-200 (thinking on, T=0.3) | 196/200 (98.0 %) |
| needles at 200K tokens | 3/3 |
| tool call | PASS |
