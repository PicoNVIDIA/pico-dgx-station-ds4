# Profiles

Each file holds the lines appended to the engine's `policy.env` for one boot, and names the number of experts to keep in GPU memory per layer. Build the matching expert map with `harness/cells/mkhot.sh <count>` before the boot.

| File | Experts on the GPU per layer | Max context | Context cache | What it is for |
|---|---:|---:|---:|---|
| `1m.env` | 291 | 1M | 1.49M tokens | Default. Full context |
| `524k.env` | 298 | 524K | 0.93M tokens | Middle step |
| `262k.env` | 302 | 262K | 0.52M tokens | Speed |
| `prefill-262k.env` | 296 | 262K | 0.37M tokens | Prefill first: 56.1K to 56.7K tok/s on long prompts, decode 2 to 3% slower |
| `chat-524k.env` | 285 | 524K | 2.25M tokens | One user on mixed prompts: four scratch areas. +5.7% over one area at the same 285 experts, +1% over `524k.env` |
| `capacity-1m.env` | 291 | 1M | 2.78M tokens | Experimental. Packed scale banks: in its one session 2 to 5% slower than `1m.env` at 2 to 32 users and 6 to 10% lower on the code prompt, with 1.87 times the context cache |
| `capacity-524k.env` | 298 | 524K | 2.06M tokens | Experimental. Packed scale banks: 2 to 5% slower than `524k.env` at 2 to 32 users, 2.2 times its context cache |

All of them gave the same output as the public recipe on the 40-prompt check at one user.

Each file starts with two comment lines. Only the `NAME=value` lines go into `policy.env`, because the launcher passes every line of that file to `env`:

```bash
grep -v '^#' configs/1m.env >> policy.env
```

## The settings

| Line | Meaning |
|---|---|
| `DS41_UTIL=0.98` | Share of GPU memory the engine may use. 0.99 ran out of memory at boot |
| `DS41_MAXLEN=...` | Maximum context in tokens. Unset means 1,048,576 |
| `DS41_GT_STAGE_SKIP=1` | Turns on the stage skip (`patches/k6-stage-skip`) |
| `DS41_GT_SCRATCH_LANES=4` | Number of scratch areas for copied experts, with layer l using area l mod N. Default 1 |
| `DS41_BATCHED_TOKENS`, `DS41_LONG_PREFILL` | Prefill chunk size (`patches/k1-knobs`) |
| `DS41_GT_SF_PACK=a2b6` | Packed 6-bit scale banks (`patches/k7-packed-scales`). Needs the stage skip |

The capacity profiles rest on a patch of about 1,600 lines that has passed its byte audit on the box but has not been reviewed line by line. Treat them as experimental.
