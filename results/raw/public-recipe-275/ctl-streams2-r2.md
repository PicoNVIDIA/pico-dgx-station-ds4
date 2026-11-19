# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-18T19:21:47+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 570 | 284.8 | 3.621 | 60 |
| 4 | 793 | 198.3 | 3.699 | 60 |
| 8 | 1,096 | 137.0 | 3.062 | 60 |
| 16 | 1,486 | 92.8 | 3.076 | 60 |
| 32 | 1,942 | 60.7 | 3.064 | 60 |
