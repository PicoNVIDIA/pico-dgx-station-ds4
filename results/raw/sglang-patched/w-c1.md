# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T05:09:46+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| GSM8K decode, T=1.0, thinking on (60 s window) | 124.8 | — | n/a | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 127.3 | — | n/a | — | window |
