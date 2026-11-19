# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T21:57:00+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|

## Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 2 | 172 | 85.9 | 2.904 | 60 |
| 4 | 210 | 52.6 | 2.771 | 60 |
| 8 | 283 | 35.4 | 2.822 | 60 |
| 16 | 402 | 25.1 | 2.887 | 60 |
| 32 | 502 | 15.7 | 2.823 | 60 |
