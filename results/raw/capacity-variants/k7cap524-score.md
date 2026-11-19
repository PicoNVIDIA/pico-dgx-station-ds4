# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T10:22:58+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 541.4 (527.9-543.0) | 124 (122-129) | 5.50 (5.42-5.66) | 98.7 (96.0-100.8) | 5 |
| coding prompt (2,048 out, greedy, thinking off) | 599.2 (553.9-618.7) | 165 (163-171) | 6.21 (5.87-6.39) | 96.6 (94.4-97.0) | 5 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 399.5 | — | 3.735 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 494.5 | — | 4.706 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 593 | 296.4 | 3.704 | 60 |
| 4 | 844 | 210.9 | 3.703 | 60 |
| 8 | 1,174 | 146.7 | 3.034 | 60 |
| 16 | 1,632 | 102.0 | 3.018 | 60 |
| 32 | 2,153 | 67.3 | 3.062 | 60 |

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14250 (14226-14274) | 281 (280-321) | 50609 (45003-50826) |
| 56805 (56758-56808) | 1068 (1056-1110) | 53101 (51218-53738) |
| 113679 (113663-113700) | 2164 (2162-2171) | 52520 (52347-52581) |

## Long-context capacity

GPU KV cache: ? tokens. 16 distinct ~110,000-token contexts decoding 2,500 forced tokens each, plus one needle request: not all ran at once (max running 16.0, max waiting 13.0), 0 preemptions, peak KV use 60 %, wall 66.1 s, all requests correct.

## Quality (same session)

| check | result |
|---|---|
| GSM8K-200 (thinking on, T=0.3) | 195/200 (97.5 %) |
| needles at 200K tokens | 3/3 |
| tool call | PASS |
