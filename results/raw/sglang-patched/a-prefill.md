# ds41 bench --full

Image `?`, driver ?, GPUs ?, util ?, rowmap `?`, recipe commit `?`, GPU KV cache ? tokens. One server session; started 2026-11-19T04:55:29+0000. Medians with the interquartile range in brackets.

## Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|

## Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14247 (14226-14284) | 3043 (2038-3444) | 4706 (4205-9233) |
| 56859 (56832-56890) | 3972 (3966-7537) | 14329 (9726-14339) |
| 113685 (113626-113764) | 7931 (7921-7932) | 14352 (14343-14354) |
