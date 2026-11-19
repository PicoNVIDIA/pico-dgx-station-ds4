# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T05:12:26+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 210 | 104.9 | n/a | 60 |
| 4 | 299 | 74.7 | n/a | 60 |
| 8 | 377 | 47.2 | n/a | 60 |
| 16 | 486 | 30.4 | n/a | 60 |
| 32 | 543 | 17.0 | n/a | 60 |
