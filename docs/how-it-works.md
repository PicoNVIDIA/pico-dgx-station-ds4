# How it works

This is the short version. The version with source citations, written for engineers who maintain offloading in vLLM and SGLang, is [`offloading.md`](offloading.md).

## The constraint

| Part | Size | Source |
|---|---|---|
| Checkpoint on disk | 510 GB | download |
| GPU memory on one GB300 | 250.7 GiB (256,703 MiB) | `nvidia-smi` |
| Routed experts (40 layers x 384 experts, 4-bit) | 253.1 GiB | arithmetic on the tensor shapes |
| Block scales for those experts | 15.82 GiB | boot log |
| Engram tables | 2 x 94.42 GiB | boot log |
| CPU memory | 494 GiB | `free` |

The routed experts alone are larger than the GPU. So the engine keeps a "hot" set of each layer's experts in GPU memory and leaves the rest, the "cold" ones, in pinned CPU memory. Each token picks 6 experts per layer. When it picks a cold one, that expert's weights are copied into a scratch area on the GPU before the layer's expert kernel runs.

That copy sits on the step's critical path. Almost everything that made this recipe faster makes those copies rarer.

## Where the memory goes at the default profile

From the boot log of the default profile (291 experts per layer, 1M context). The boot excerpt is in `results/logs/boot-excerpt-1m.log`.

| What | Where | Size |
|---|---|---|
| Hot experts, 291 per layer, plus the shared scratch | GPU | 193.4 GiB, of which the scratch is 1.53 GiB |
| Block scales of all 384 experts per layer | GPU | 15.82 GiB |
| Cold experts, 93 per layer | CPU, pinned | 61.3 GiB |
| Engram tables | CPU, pinned | 189.1 GiB |
| Context cache overflow tier | CPU | 48 GiB |
| Context cache | GPU | 1,488,297 tokens |

Everything else on the GPU (attention and dense weights, the draft model, the output head, working memory) is not itemised in the boot log. The upstream author's published total suggests about 20 GiB for the weights among them; we have not measured that split.

## What the public recipe already does

Garner McCloud's engine is a set of fused components and kernels for this model, loaded over the stock vLLM v0.31.0 container. James Meadlock's expert tier ("gracetier") adds the hot and cold split for a single GPU, and a manager that counts which experts are picked and exchanges hot and cold experts between steps to follow the traffic. On this box that public recipe runs at 2.1 to 2.9 times stock vLLM for one user, depending on the prompts, and 3.9 times at 32 users.

Garner McCloud's fastest published recipe uses a second GPU card (an RTX PRO 6000) to compute the cold experts. That removes the copies. The Station we measured on has no such card.

## What this repo adds

### 1. The port to a Station without the second card

The upstream launcher assumes the second card. `harness/setup/station_gt_prep.sh` builds a directory shim and patches a private copy of the upstream launch card (`harness/setup/card-gt1-station.diff`, 42 lines) so the one-GPU tier boots on the GB300 alone.

At the published setting of 255 experts per layer, the first boot gave 263.7 and 263.0 tok/s on James Meadlock's two prompt sets (`results/logs/gt1-first-boot-card.log`, lines 34 and 35). He publishes 261.5 and 260.5, measured on a Station that has the second card; his README says a Station without it was not tested.

### 2. More experts in GPU memory

| Setting | Public recipe | This recipe |
|---|---|---|
| GPU memory utilization | 95% | 98% (99% runs out of memory at boot) |
| Hot experts per layer, 1M context | 275 | 291 |
| Hot experts per layer, 524K context | | 298 |
| Hot experts per layer, 262K context | | 302 |

Two sources of memory pay for the extra experts: the utilization the launcher left unused, and a shorter maximum context. One more hot expert in every layer costs 0.659 GiB. A 1M-token context needs 2.75 GiB of context cache before the first request can be served.

The cost is context cache. At 1M the cache drops from 2.71M tokens to 1.49M.

Which experts are hot at boot is not a lever. The helper `harness/cells/mkhot.sh` adds the lowest-numbered cold experts to the upstream 275 map; the manager then repairs that starting set during warm-up. A boot from a map fitted to the traffic measured the same once warm.

### 3. The stage skip

The scratch is one area shared by all 40 layers, with one slot for each cold expert position (93 slots at 291 hot). Cold expert number c of any layer always lands in slot c. Stock gracetier copies the expert in on every call that picks it.

The patch in `patches/k6-stage-skip` keeps a tag per slot: which layer's expert it holds now. If a call needs an expert whose tag is still in place, the copy is skipped. A slot stays valid for a layer only until another layer copies into the same slot, and a residency exchange resets the tags. The decision is made on the GPU inside the existing copy kernel, so no launch is added.

- Hit rate: about 25% of copies for one user. A replay of a copy log predicts about 5% at 5 to 8 users and 2% at 17 to 32 (`results/analysis/stage-log-analysis-285.txt`). Counters over whole sessions that mix user counts read 7 to 20%.
- Gain: 4.5 to 6.0% for one user on the two mixed prompt sets, with the patch switched on against off on the default profile (`sk1_1m`, `sk0_1m`). At 8 and 32 users, 1% or less. On the code prompt the difference (4%) is inside the spread between boots. An earlier pair at 285 experts gave 4.7% and 5.2% on the mixed sets.
- Safety: an audit mode copies anyway and compares the bytes. 217,623 skips were compared in one-user traffic with 0 differing words, and 1,118 sampled skips at 16 users with 0 differing. Over the 3.5-hour soak at 2 to 32 users, 14,526 sampled skips, 0 differing.
- The option `DS41_GT_SCRATCH_LANES=N` gives N scratch areas, with layer l using area l mod N, which raises the hit rate at the cost of GPU memory.

The patch is off unless `DS41_GT_STAGE_SKIP=1` is set.

## Where the time goes now

One step for one user on the fixed code prompt, profiled with Nsight Systems at 298 hot experts and a 524K context (`results/profiles/profc-table.txt`). The GPU is 99% busy, so the time is in the kernels and not in gaps between them.

| Part of the step | Share |
|---|---|
| Expert math (one fused kernel per layer) | 41% |
| Attention | 25% |
| Mixes and seams between layers | 10% |
| Copying cold experts | 10% |
| Draft model | 6% |
| Output head | 4% |
| Idle | 4% |

On a prompt with almost no cold picks the step is 8.0 ms (`results/profiles/profg-table.txt`); on the code prompt it is 9.7 ms.

By our estimate the expert kernel is within about 20% of what the GPU's memory bandwidth allows: it measures 94 microseconds a layer, and reading the bytes it touches at the published bandwidth would take about 75. That is arithmetic on a datasheet figure, not a measurement. If it holds, what is left is new kernel work on the other 60% of the step.
