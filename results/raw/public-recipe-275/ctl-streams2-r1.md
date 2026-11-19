# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-18T19:15:06+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 554 | 277.2 | 3.693 | 60 |
| 4 | 787 | 196.8 | 3.706 | 60 |
| 8 | 1,104 | 138.0 | 3.067 | 60 |
| 16 | 1,478 | 92.4 | 3.05 | 60 |
| 32 | 1,951 | 61.0 | 3.063 | 60 |
