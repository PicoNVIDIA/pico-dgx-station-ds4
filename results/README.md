# Results

`summary.json` holds every number used in the README and the charts, with the directory of raw files behind each column.

| Directory | What is in it |
|---|---|
| `raw/` | Benchmark scoreboards as the benchmark wrote them on the box (`.md` and `.json`) |
| `logs/` | Run logs of the measurement queues, gate verdicts, probe logs, the other engines' logs, one boot excerpt |
| `profiles/` | Step profiles (Nsight Systems tables) |
| `identity/` | Output captures for the 40 prompts and the comparison with stock vLLM |
| `analysis/` | The analysis of a copy log: which cold experts each step copies, by number of users |

## Raw scoreboards

| Directory in `raw/` | Column | Files |
|---|---|---|
| `stock-vllm/` | Stock vLLM v0.32.0 | one benchmark session (one user, 8 and 32 users) |
| `sglang-patched/` | SGLang nightly with one line changed | `w-*`: code prompt and stream windows counted at the client, two passes. `a-*`: coding prompt, prefill, quality |
| `public-recipe-275/` | Public one-GPU recipe on our box | `score-gt275`: the full scoreboard. `ctl-streams2-r1`, `-r2`: the two warmed stream passes behind the stream rows |
| `this-recipe-1m/` | This recipe, 1M context | full scoreboard |
| `this-recipe-524k/` | This recipe, 524K context | full scoreboard |
| `this-recipe-262k/` | This recipe, 262K context | full scoreboard without the capacity stage (the context cache is too small for it) |
| `capacity-variants/` | Packed scale banks at 1M and 524K | full scoreboards |

## Logs, profiles and identity

`logs/gtqN-run.log` are the run logs of measurement queues 3 to 22 as the runner wrote them on the box (queue 17 only booted a server and is not included). The cell name in a log line is the config; its settings are in `harness/cells/defs/<cell>.env`.

| Claim | Where to look |
|---|---|
| Same output on 40 of 40 prompts for a config | `grep "vs reference" logs/*.log` (one line per boot) |
| The public recipe at its published 255 experts reproduces on this box: 263.7 and 263.0 tok/s | `logs/gt1-first-boot-card.log`, lines 34 and 35 |
| One user on the 40 mixed prompts | lines `<cell>-c1-r1-a` and `-b` in the run logs: `ctl` (public recipe, queue 4), `f1m`, `f524`, `f262` (queue 8). Stock vLLM: `logs/stock-vllm-bench-run.log` |
| Skip patch byte audit: 217,623 skips, 0 differing words | `logs/gtq7-run.log` (the last `stage_skip` event), verdict in `logs/gtq7-gate.log` |
| 10-minute mixed-load soak of the speed profile, 1,118 sampled skips | `logs/gtq12-run.log`, cell `soak262` |
| 3.5-hour soak of the default profile, 14,526 sampled skips | `logs/gtq21-run.log`, cell `soak1m`. One sample a minute in `logs/gtq21-soak-mem.csv`: time (UTC), GPU MiB used, host MiB used, host MiB available, container restarts |
| Prompts the server had not seen (held-out test) | `logs/gtq22-run.log`, cell `ho1m` |
| Skip patch switched off against on, default profile | `logs/gtq22-run.log`, cells `sk0_1m` and `sk1_1m` |
| Short-request bursts | `logs/gtq12-run.log`, cell `a5f262` |
| Draft-count sweeps, extra scratch areas, prefill chunk size | `logs/gtq4-run.log` to `logs/gtq13-run.log` |
| Packed scales: four gates, then speed | `logs/gtq18-run.log`, `logs/gtq19-gate.log`, `logs/gtq19-run.log`, `logs/gtq20-run.log` |
| The GPU's 4-bit and 6-bit unpack behaviour | `logs/k7-e2.log` |
| Scale-shift arithmetic is exact, with controls | `logs/k7-e1.log` |
| Stock SGLang does not load the model on one GB300 | `logs/sglang-probe-run.log` and the three `logs/sglang-probe-*-tail.log` files |
| SGLang with one line changed: boots, speeds, quality | `logs/sglang-patched-pass2-run.log`, `logs/sglang-patched-pass3-run.log`, `raw/sglang-patched/` |
| The client-side token count equals the server counter on this engine | `logs/gtq15-run.log` |
| Memory map of the default profile | `logs/boot-excerpt-1m.log` |
| Where the time goes in a step | `profiles/profc-table.txt` (code prompt), `profiles/profg-table.txt` (general prompt) |
| Cost of the packed-scale kernel | `profiles/profk0-table.txt`, `profkb-table.txt`, `profkp-table.txt` |
| Output against stock vLLM | `identity/stock-vs-engine.txt`; the captures are in `identity/captures/` |
| Cold experts copied per layer call, by number of users | `analysis/stage-log-analysis-285.txt` |

Cells that failed to boot are in these logs too (`NOT SERVING`, `FAILED_BOOT`). One hardware identifier, the GPU's UUID, is redacted in `logs/gt1-first-boot-card.log`; nothing else in the logs was edited.

To re-check an identity result from the captures: `python3 harness/identity/ident_cmp.py results/identity/captures/gt275-run1.json results/identity/captures/gtq8-f1m-c1.json`.

## Reading them

- The benchmark's header line shows `?` for image, driver and commit: our runs did not pass those fields to it. The values are in `summary.json` under `meta` and in `PINS.md`.
- Stream rows in `score-gt275` are the first stream pass after that boot and read 0 to 3.5% under the two later passes. The public recipe's stream numbers in the README are the mean of those two later passes.
- A range in the README is the lowest and highest value across boots or passes of one config. The single value in a scoreboard file is one session.
- The stock vLLM stream rows were run once. They read 14% under the published numbers for that config (326 at 8 users, 584 at 32).
- The benchmark labels its capacity contexts 110K and its needle prompt 200K. By the token counts in the raw files they are 95K and 174K.
- The two-GPU reference numbers in `summary.json` are Garner McCloud's published figures. We did not measure them.
