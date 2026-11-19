# What we tried

Everything measured on one Station on 18 and 19 November 2026. "Same output" means the 40-prompt greedy check at one user matched the reference token for token. Each row is one boot against one boot unless it says otherwise. The run logs are in `results/logs/`; the cell name in brackets finds the lines.

## Adopted

| Change | Result | Output |
|---|---|---|
| GPU memory utilization 95% to 98%, more hot experts (275 to 291 at 1M) | +5.2 to +10.5% from 2 to 32 users; +10.6% on the code prompt with both taken first in a session | same |
| Shorter maximum context for more hot experts (298 at 524K, 302 at 262K) | 2 to 4% per step at many users in single sessions; the second step is close to the spread between boots | same |
| Stage skip [`sk1_1m` against `sk0_1m`; earlier `k6b` against `h285u97`] | +4.5 to +6.0% for one user on the two mixed prompt sets on the default profile (+4.7% and +5.2% in the earlier pair at 285 experts). 1% or less at 8 and 32 users | same |

## Optional variants

| Change | Result | Why it is not the default |
|---|---|---|
| Four scratch areas [`k6l4`] | For one user on mixed prompts: +5.7% over one scratch area at the same 285 experts, and +1% over the 524K profile, which spends that memory on 13 more experts | nothing at many users or on the code prompt |
| 16,384-token prefill chunks [`pf16kf`] | prefill 56.1K to 56.7K tok/s, about +5% | decode is 2 to 3% slower, costs about 4.4 GiB |
| Packed 6-bit scale banks [`k7cap1m`, `k7cap524`] | frees 3.9 GiB: context cache grows by 1.1M to 1.3M tokens | 2 to 5% slower at 2 to 32 users and up to 10% lower on the code prompt, one session each; experimental patch |

## Tried, no gain

| Change | Result |
|---|---|
| Fewer or more draft tokens at 5 to 32 users (1, 2, 4 or 5 against the shipped 3) | no gain: at 32 users 1,708 to 1,915 against 1,942 to 1,951 (4 drafts ties at 16 users) |
| One-user draft count fixed at 7 [`k1f7f`] | 5% slower on mixed prompts |
| One-user draft count fixed at 3 [`k1f3f`] | 445 tok/s on the code prompt against 546 to 581, and the output changes (4 of 40 match) |
| GPU memory utilization 99% | out of memory at boot |
| Two scratch areas at 1M [`k6l2`] | +2% on one prompt set, costs 0.71M tokens of context cache |
| Faster swapping of the hot set (live residency settings) | +1% |
| Starting from a hot set fitted to the traffic [`fit302`] | no gain once warm: the manager repairs the starting set in the warm-up rounds |
| `tail_drafthead_mxfp8` from the two-GPU recipe [`ord_dh`] | +1.2% on mixed prompts, costs context cache, not adopted |
| `w8oproj` from the two-GPU recipe [`ord_w8`] | changes the output (4 of 40 match). The step is no shorter; tok/s is +1% and +4% on the two sets because the different text drafts differently |
| `lowb_aux` from the two-GPU recipe | refused at boot together with the expert tier |
| Packed scale banks used for more hot experts [`k7p307`, `k7p308`, `k7p296m`] | net 1 to 2% slower: the packed read costs 3%, five more experts buy back 1.5% |
| 4-bit packed scales | not possible this way: the GPU's 4-bit unpack multiplies each value by four (`results/logs/k7-e2.log`) |
| Sharing identical memory pages between scale banks | no identical pages exist (`results/logs/k7-survey.log`) |

## What the profiles say

- The GPU is 99% busy during a step. There is no idle gap to remove.
- A step with almost no cold copies takes about 8.0 ms. Mixed traffic runs at 9.0 ms and the code prompt at 9.7 ms.
- The expert kernel is about 40% of a step.
- An extra hot expert per layer is now worth about 0.3%, so this lever has flattened.

## Other engines on the same box

| Engine | Result | Evidence |
|---|---|---|
| Stock vLLM v0.32.0, its own Station config | 167.8 to 177.8 tok/s for one user on the code prompt, 504 at 32 users (two boots, 24 requests at once). GSM8K-200 193 of 200 | `results/raw/stock-vllm/` |
| Stock SGLang (nightly `6b374a55`) | does not load: its CPU-memory expert tier raises `MXFP4 Marlin requires SM90 or SM120` after loading for eight minutes, and generic CPU offload runs out of GPU memory. Its four-GPU kernel path crashed in argument checking at one GPU | `results/logs/sglang-probe-run.log` and the three `sglang-probe-*-tail.log` files |
| SGLang with that one check removed | loads without the draft model: 135.6 tok/s for one user, 552 at 32 users, GSM8K-200 195 of 200 | `results/raw/sglang-patched/`, `harness/sglang/sglang_e2_entry.sh` |
