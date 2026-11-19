# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T05:19:07+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 206 | 102.9 | n/a | 60 |
| 4 | 296 | 74.0 | n/a | 60 |
| 8 | 379 | 47.4 | n/a | 60 |
| 16 | 457 | 28.5 | n/a | 60 |
| 32 | 561 | 17.5 | n/a | 60 |
