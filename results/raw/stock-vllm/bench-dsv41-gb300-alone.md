# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-18T17:35:54+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 177.8 (174.1-178.0) | 234 (234-239) | 3.60 (3.52-3.60) | 49.5 (49.4-49.5) | 5 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 134.5 | — | 2.789 | — | window |

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 8 | 281 | 35.2 | 2.833 | 60 |
| 32 | 501 | 15.7 | 2.805 | 60 |
