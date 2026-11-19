# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T22:03:41+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 163 | 81.3 | 2.675 | 60 |
| 4 | 217 | 54.2 | 2.879 | 60 |
| 8 | 274 | 34.2 | 2.762 | 60 |
| 16 | 403 | 25.2 | 2.837 | 60 |
| 32 | 506 | 15.8 | 2.831 | 60 |
