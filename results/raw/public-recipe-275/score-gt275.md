# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-18T18:54:28+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 510.5 (495.4-515.9) | 142 (138-147) | 5.42 (5.22-5.43) | 95.0 (90.7-97.9) | 5 |
| coding prompt (2,048 out, greedy, thinking off) | 568.1 (503.8-586.1) | 190 (186-197) | 6.44 (6.39-6.44) | 87.7 (79.2-91.0) | 5 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 396.1 | — | 3.814 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 486.9 | — | 4.696 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 562 | 281.2 | 3.691 | 60 |
| 4 | 776 | 194.0 | 3.588 | 60 |
| 8 | 1,086 | 135.7 | 3.025 | 60 |
| 16 | 1,452 | 90.8 | 3.045 | 60 |
| 32 | 1,878 | 58.7 | 3.079 | 60 |

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14225 (14212-14234) | 330 (316-1252) | 43190 (24866-45133) |
| 56886 (56832-56922) | 1142 (1126-2029) | 49728 (34620-50534) |
| 113585 (113566-113602) | 2296 (2291-2302) | 49477 (49348-49592) |

## Long-context capacity

GPU KV cache: ? tokens. 16 distinct ~110,000-token contexts decoding 2,500 forced tokens each, plus one needle request: all ran at once (max running 17.0, max waiting 13.0), 0 preemptions, peak KV use 63 %, wall 75.2 s, all requests correct.

## Quality (same session)

| check | result |
|---|---|
| GSM8K-200 (thinking on, T=0.3) | 195/200 (97.5 %) |
| needles at 200K tokens | 3/3 |
| tool call | PASS |
