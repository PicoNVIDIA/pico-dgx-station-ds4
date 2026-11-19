# One GB300, a model larger than its memory: how this recipe tiers experts, next to vLLM UVA offload and SGLang Paged Experts

For the engineers who maintain vLLM and SGLang offloading for this machine. Written 19 November 2026 from the sources listed at the end.

**How to read it.** Citations are `(ROOT path:lines)`; the roots are listed under [Source roots](#source-roots) at the end. Number labels: **[M]** measured by us on one Station on 18 and 19 November 2026, **[S]** read from source, **[P]** published by the upstream author, **[E]** our estimate (arithmetic on source constants unless stated). "inferred" marks a conclusion we drew and did not read. UNVERIFIED marks what we could not check from the files we have.

**Whose work this is.** The engine (fused components, kernels, launcher) is Garner McCloud's. The one-GPU expert tier, gracetier, with its residency manager is James Meadlock's. Ours: three patch sets and the hot counts.

## 1. Summary

1. The routed experts are 253.1 GiB [E]. The GPU reports 250.7 GiB [M] (OURS PINS.md:20). So each of the 40 layers keeps 291 to 302 of its 384 experts in GPU memory and the rest in pinned CPU memory. The unit of residency is one whole expert.
2. When a token picks a cold expert, one kernel per layer copies that expert's two weight matrices (17.7 MB [E]) from CPU memory into a fixed GPU scratch slot. For decode calls of up to 16 tokens this is inside the captured graph. One expert kernel then runs over all 384 groups. Block scales for all 384 experts stay on the GPU.
3. The same kernel counts picks. Between engine steps a host-side manager exchanges a hot expert with a cold one at fixed addresses. No graph is recaptured.
4. Measured once: 540 to 565 tok/s for one user and 2,148 at 32 users, at a 1M context [M]. Stock vLLM with UVA offload: 167.8 to 177.8 and 504, over two boots. SGLang Paged Experts with one line changed: 135.6 and 552 [M].
5. None of this is a vLLM flag. It is Python loaded at interpreter start over the stock v0.31.0 image, 12 overlay files, 9 prebuilt kernel libraries and a forked pair of DeepGEMM headers. Output matches the public recipe at one user. It does not match stock vLLM.

## 2. What is kept where

| Tensor group | Where it lives in this recipe | Size | Citation |
|---|---|---|---|
| Hot routed experts, H per layer | GPU. One private VMM allocation per layer and per weight tensor, groups 0..H-1. vLLM builds its module for H experts only; gracetier copies the finalized bytes in | 191.8 GiB at H=291, 196.4 at 298, 199.1 at 302 [E]. The boot log gives 193.4 GiB at H=291 for hot groups plus scratch [M] | (GT tree/fastpath/fastpath_gracetier.py:896-909, 663-670) |
| Cold routed experts, 384-H per layer | CPU. One anonymous huge-page mapping, registered with `cudaHostRegister`. Weights only, already in the expert kernel's layout. Hot experts are not in it | 61.3 GiB at H=291 [M, boot log] | (GT tree/fastpath/gracetier_rt/runtime/gracetier_pool.py:1-14, 191-198) (OURS results/logs/boot-excerpt-1m.log) |
| Scratch slots | GPU. One physical allocation per weight tensor kind, mapped behind every layer's hot block. 384-H slots, shared by all 40 layers | 1.53 GiB at H=291 [M, boot log]. Private boundary granules: at most 0.16 GiB [P] | (GT tree/fastpath/fastpath_gracetier.py:502-517) (GT tree/fastpath/gracetier_rt/runtime/gracetier_vmm.py:4-16) (DEC docs/P1-GRACETIER-DESIGN.md:242) |
| Block scales, all 384 x 40 experts | GPU. Garner McCloud's `a12_retire` moves them into a generic-compression allocation at load | 15.82 GiB [M, boot log]. Of that, 3.8 GiB is for cold experts at H=291 [E] | (GT tree/fastpath/gracetier_rt/runtime/gracetier_vmm.py:176) (GT tree/fastpath/fastpath_a12_retire.py:1-8) |
| Engram tables, two | CPU, pinned, by vLLM's own `--engram-config '{"cpu_offload": true}'` | 2 x 94.42 GiB [M, boot log] | (GT tree/serve/serve.sh:341) (OURS results/logs/boot-excerpt-1m.log) |
| Attention, dense, shared expert, gate, output head | GPU | About 20 GiB together with the draft model [E, from a published total] | (GT tree/docs/ARCHITECTURE.md:48) |
| Draft model (DSpark) | GPU. Its three expert layers have 128 experts each and are not tiered | `mtp.*` 7.4 GiB [P] | (GT tree/fastpath/fastpath_fastload.py:3-5) (GT tree/fastpath/fastpath_a12_retire.py:29-30) |
| KV cache | GPU pool in `fp8_ds_mla`, plus a 48 GB host overflow tier through vLLM's offloading connector | 1.49M tokens at a 1M context [M]. One 1M request needs 2.75 GiB [M] | (GT policy.env:7-8) (GT tree/serve/serve.sh:333, 342) |

Per expert the two matrices are 11,796,480 B and 5,898,240 B [S] (GT tree/fastpath/gracetier_rt/runtime/gracetier_pool.py:124-129). One more hot expert in every layer is 0.659 GiB, by that arithmetic and as measured [M]. Garner McCloud's load note gives 91.6 GiB per Engram table (GT tree/fastpath/fastpath_fastload.py:3-4); our boot log gives 94.42.

## 3. What happens when a token picks a cold expert

1. **Router.** One launch does the gate product and top-6, maps each picked expert through the layer's row map, and writes the result into the expert kernel's index buffer (GT tree/fastpath/fastpath_boundary.py:1367-1373) (GT tree/kernels/w4-boundary/src/csrc/router.cu:526-536). In gracetier's map every entry is a group number: 0..H-1 for hot, H+c for the cold expert in slot c (GT tree/fastpath/fastpath_gracetier.py:24-27). The router does not treat cold picks differently. No host code runs.
2. **Decision.** `gt_stage` is one Triton launch per layer call, with a grid of (cold slots, 48 programs). Program (c, q) scans the call's group ids on the device for H+c and exits if the slot was not picked (GT tree/fastpath/gracetier_rt/runtime/gracetier_stage.py:149-162).
3. **What is copied.** The whole expert's weights, both matrices, in 8 KiB blocks striped over the 48 programs. Scales are not copied (GT tree/fastpath/gracetier_rt/runtime/gracetier_stage.py:9-12, 55-56, 166-180).
4. **From where.** The expert's fixed slot in the host pool. The pool is registered in chunks of at most 64 GiB with the Portable and Mapped flags. The kernel loads from the host address, which under UVA is the device address (GT tree/fastpath/gracetier_rt/runtime/host_tier.py:1-15, 158-179) (GT tree/fastpath/gracetier_rt/runtime/gracetier_pool.py:226-237).
5. **To where.** Group H+c of the layer's weight tensor. Each layer's tensor is one contiguous address range built with `cuMemAddressReserve` and `cuMemMap`: H private groups, then the shared scratch (GT tree/fastpath/gracetier_rt/runtime/gracetier_vmm.py:395-419). Sharing is safe because layers run in stream order and a layer reads only the cold groups its own call staged (GT tree/fastpath/gracetier_rt/runtime/gracetier_vmm.py:12-14).
6. **Expert kernel.** DeepGEMM's MegaMoE, compiled from Garner McCloud's fork, runs once over 384 groups and reads hot and staged groups the same way. Its own combine sums the six routes (GT tree/fastpath/fastpath_boundary.py:1385-1391).
7. **Graph capture.** For decode calls of up to 16 tokens, steps 1 to 6 are inside the captured graph: the copy picks its sources on the device and every address is fixed before capture (GT tree/fastpath/fastpath_boundary.py:3, 137) (GT tree/fastpath/fastpath_gracetier.py:35-36). Larger calls go through a wrap of the module's forward: a row-map lookup, `gt_stage`, then vLLM's stock forward (GT tree/fastpath/fastpath_gracetier.py:936-952). A request adds one row plus its draft tokens, so steps for three or more users, and 32-row copy steps for one user, exceed 16 tokens, inferred (GT tree/serve/serve.sh:331) (GT tree/fastpath/fastpath_w9_suffix.py:8-12). Whether they are captured is UNVERIFIED.
8. **Cost.** 46 to 50 microseconds per cold expert [P] (GT tree/fastpath/fastpath_gracetier.py:5), about 370 GB/s [E]. The copy is serial: the expert kernel waits. In our profile it is 40 calls of 27 microseconds, 9.7 percent of the step [M] (OURS results/profiles/profc-table.txt:22, 33). A layer call stages 1.28 cold experts on average at one user and 6.63 at 17 to 32 users [M, H=285] (OURS results/analysis/stage-log-analysis-285.txt:26, 46).

**What the stage skip changes.** Our patch adds a variant of the same kernel. A tag table holds one word per program per scratch slot. The key is layer x 384 plus the expert now in that slot. If tag equals key, the program does not copy. Otherwise it copies as before and stores the key (OURS patches/k6-stage-skip/0001-gracetier-stage-skip.patch:364-378, 398-417). The launch count is unchanged. Because the scratch is shared, a slot stays valid for a layer only until another layer stages into it, and every layer must have the same H (OURS patches/k6-stage-skip/0001-gracetier-stage-skip.patch:94-97). A residency swap resets the slot's tags (OURS patches/k6-stage-skip/0001-gracetier-stage-skip.patch:187-193, 348-349). About 25 percent of fetches are skipped for one user, worth about 5 percent on mixed prompts [M]. At 17 to 32 users a replay of the staging log gave 1.7 percent [M] (OURS results/analysis/stage-log-analysis-285.txt:53). An audit build compared 217,623 skips word for word and found 0 differences [M].

## 4. How the hot set is chosen and how it adapts

**Row map.** Boot reads a JSON file with a hot list, a cold list and an expert-to-group row per layer. It refuses anything that is not a partition (GT tree/fastpath/fastpath_gracetier.py:168-191). The shipped file starts from Garner McCloud's 255-expert map, built from measured route counts (GT tree/docs/ARCHITECTURE.md:36-37), extended to 275 by James Meadlock (GT tree/fastpath/hook/rowmap-allres.json:1) (JM deepseek-v4.1-flash-vllm-gracetier/README.md:98). Our helper adds the lowest-numbered cold experts to reach 291, 298 or 302. That choice only seeds the manager (OURS harness/cells/mkhot.sh:3-4, 14-20).

**Counters.** For calls of at most 256 tokens, program (0, 0) of `gt_stage` adds one per pick, hot or cold, to an int32 [40, 384] table indexed by expert, with relaxed atomics (GT tree/fastpath/gracetier_rt/runtime/gracetier_stage.py:60, 196-202) (GT tree/fastpath/fastpath_gracetier.py:644-647).

**Policy.** Every 64 engine steps the table is copied to pinned host memory without blocking. A later step folds the delta into a decayed count with a half-life of 500 steps. Per layer it pairs the most-used cold experts with the least-used hot ones, at most two pairs. A pair is queued only if the cold count is at least 3.0 times the hot count and at least 8 (GT tree/fastpath/gracetier_rt/runtime/gracetier_res.py:81-91, 407-420) (GT tree/fastpath/gracetier_rt/runtime/residency_policy.py:209-263) (GT tree/fastpath/gracetier-res.json:1).

**When swaps happen.** gracetier wraps `GPUModelRunner.execute_model`. The manager's barrier runs first, on the step's thread and stream, and applies at most four queued swaps. During graph capture it returns at once (GT tree/fastpath/fastpath_gracetier.py:868-880) (GT tree/fastpath/gracetier_rt/runtime/gracetier_res.py:387-405).

**A swap** is five operations queued between two steps: hot group to scratch slot, host slot to hot group, scratch slot to host slot, an exchange of the two scale rows in both banks, then the row map and its inverse (GT tree/fastpath/gracetier_rt/runtime/gracetier_res.py:5-17, 476-490).

**Why output stays the same.** Bytes move and the row map moves with them, in stream order with the steps, so each pick still reads its own expert. A check raises if a step runs on a stream other than its barrier's (GT tree/fastpath/gracetier_rt/runtime/gracetier_res.py:365-375). Boot digests of every expert back an on-demand audit (GT tree/fastpath/gracetier_rt/runtime/gracetier_res.py:47-50); James Meadlock reports 0 mismatches in 8 audits [P] (JM deepseek-v4.1-flash-vllm-gracetier/README.md:136). That the result does not depend on which group holds an expert is inferred from the kernel summing in top-k column order (GT tree/fastpath/gracetier_rt/runtime/dg_mega.py:13-14).

On steady traffic the cold share of picks was 1 to 5 percent [M].

## 5. Comparison

| | This recipe | Stock vLLM v0.32.0, UVA offload | SGLang Paged Experts |
|---|---|---|---|
| Unit of residency | One expert per layer, weights only | UNVERIFIED: the offload backend's source is not on this machine. The flags name whole parameters and a byte budget (TC glm-5.3-flash-fp8/baseline/serve-deepseek-gb300-stock.sh:32). James Meadlock reports "9 UVA layers" at a budget of 54, so whole layers, inferred (JM deepseek-v4.1-flash-vllm-uva-dspark/README.md:864) | One expert per layer, in a K-slot table (SGL docs/docs/advanced_features/paged_experts.mdx:7-10) |
| Where cold weights live | Pinned CPU memory, cold experts only. A swap writes the evicted expert back | Pinned CPU memory, read in place over the CPU-GPU link every step (TC glm-5.3-flash-fp8/baseline/serve-deepseek-gb300-stock.sh:4-5) | Pinned CPU memory holding all E experts (SGL docs/docs/advanced_features/paged_experts.mdx:91-92) (SGL python/sglang/srt/layers/moe/paged_experts/store.py:34-56). Eviction then only drops the GPU copy, inferred |
| What triggers a move | A cold pick copies to scratch for that call, with no eviction. Decayed counts trigger swaps between steps | Nothing moves at run time, inferred from the same comment. UNVERIFIED | A routing miss. LRU eviction is decided by a GPU kernel (SGL python/sglang/kernels/jit/csrc/moe/paged_experts.cuh:1-11) |
| Fetch on the critical path | Yes, serial before the expert kernel. 9.7 percent of a step [M] | Yes, inferred from the same comment: the weights are read over the link every step. About 4 ms of an 11 ms step on an older image [P] (JM deepseek-v4.1-flash-vllm-uva-dspark/README.md:1035) | Yes: missing experts are copied "before the expert GEMM runs" (SGL docs/docs/advanced_features/paged_experts.mdx:9-10) |
| Draft model | Yes (GT tree/serve/serve.sh:331) | Yes (TC glm-5.3-flash-fp8/baseline/serve-deepseek-gb300-stock.sh:36) | Rejected at startup (SGL python/sglang/srt/layers/moe/paged_experts/method.py:497-498) |
| Batching | 32 sequences (GT tree/serve/serve.sh:35). Fused path up to 16 tokens per call, counting up to 256 | 24 sequences; at 32 users 8 wait (TC glm-5.3-flash-fp8/baseline/serve-deepseek-gb300-stock.sh:11, 31) | Decode graphs up to K // top_k requests. Larger steps run in waves without a graph (SGL docs/docs/advanced_features/paged_experts.mdx:141-151, 165-166) |
| GPU memory overhead of the tier | 1.5 GiB scratch plus 3.8 GiB of cold experts' scales at H=291 [E] | UNVERIFIED | The K slots only, per the docs. Scale handling for this format UNVERIFIED |
| Context cache at 1M | 1.49M tokens [M]. Public recipe: 2.71M [M] | 6 GiB set by flag, about 3.1M tokens (TC glm-5.3-flash-fp8/baseline/serve-deepseek-gb300-stock.sh:8, 33) | Run at 262K only. Size not recorded |
| What must change in the framework | See section 6 | Nothing. Flags only | One line for this GPU: the Marlin MXFP4 method raises unless SM90 or SM120 (SGL python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py:161-162) |
| One user / 32 users, tok/s [M] | 540 to 565 / 2,148 at 1M. 545 to 581 / 2,299 at 262K. Public recipe: 510.5 / 1,946 | 167.8 to 177.8 / 504. Two boots, the second warmed; 24 requests at once; the 8-user and 32-user cells read 14 percent under the numbers published for this config in both | 135.6 / 552. 255 resident, memory fraction 0.90, no draft, 262K |

The SGLang run used `--paged-experts-num-resident 255 --moe-runner-backend marlin --disable-shared-experts-fusion` (OURS harness/sglang/station_sglang_e2d.sh:32). Stock SGLang refuses or runs out of memory (OURS results/logs/sglang-probe-run.log:8-22, 67). James Meadlock reports that vLLM v0.31.0's own MegaMoE backend builds a full GPU copy of all experts, so UVA offload cannot shrink it [P] (JM deepseek-v4.1-flash-vllm-uva-dspark/README.md:704).

## 6. What this needs changed relative to stock vLLM

| Kind | What |
|---|---|
| Launch flags | Flags of the stock image's `vllm serve`: `--moe-backend deep_gemm_mega_moe`, `--engram-config`, `--speculative-config`, `--kv-offloading-size`, `--gpu-memory-utilization`, 26 capture sizes (GT tree/serve/serve.sh:331-349). One component rewrites the draft schedule inside the argument module (GT tree/fastpath/fastpath_w9_suffix.py:237-255) |
| Environment | `DS41_FASTPATH` (the component list), `MEGA_ROWMAP`, the FlashInfer tune file; with our patches `GRACETIER_STAGE_SKIP` and the `DS41_K1_*` knobs (GT tree/serve/serve.sh:316-324) (OURS patches/k6-stage-skip/serve-env-lines.txt:4-6). Residency is set by a JSON file, not a variable (GT tree/fastpath/fastpath_gracetier.py:64-67) |
| Python at start | `sitecustomize.py` is mounted over the image's. It installs the expert hook, then patches each named component's target module as it is imported (GT tree/fastpath/sitecustomize.py:13-34) (GT tree/fastpath/fastpath_hook.py:61-109) |
| Device code | Triton kernels compiled at boot, DeepGEMM's JIT fed two forked headers, and 9 libraries prebuilt with `nvcc` for sm_103a |

**Overlay files**, mounted read-only over the image's `vllm` package (GT tree/serve/serve.sh:67-77).

| Files | What each replaces | Citation |
|---|---|---|
| `config/cache.py`, `models/deepseek_v41/nvidia/model.py`, `model_state.py`, `vl_model.py`, and the new `models/deepseek_v41/decoder_replay_layers.py` | Stock files plus the runtime diff of vLLM PR 58132 (bounded replay for layers 21 to 39) | (GT tree/docker/overlay-58132/README.md:8-12) |
| `v1/worker/gpu/sample/gumbel.py`, `v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | Stock files plus backports of vLLM PRs 61161 and 60945 | (GT overlay-vllm/README.md:11-21) |
| `parser/deepseek_v41.py` | Stock file plus a tool-parser fix from James Meadlock's recipe | (GT overlay-vllm/README.md:30-42) |
| `tokenizers/deepseek_v41_encoding.py`, `tokenizers/deepseek_v41.py`, `entrypoints/anthropic/serving.py`, `renderers/deepseek_v4.py` | Prompt format of the 1017 checkpoint | (GT tree/docker/overlay-1017/README.md:11-16) |

**Fast-path components**: the 23 names in the policy file (GT policy.env:10). All are Garner McCloud's except `gracetier`. His per-component notes are in (GT tree/docs/ARCHITECTURE.md:80-106). The public one-GPU list leaves out five of his components, three because they change numerics (GT launch-gt1.sh:15-16).

| Group | Names | What it does | Files in `tree/fastpath/` |
|---|---|---|---|
| Expert tier | `gracetier`, `megamoe` | Hot and cold tier; scheduling-only fork of the expert kernel | `fastpath_gracetier.py`, `gracetier_rt/`, `fastpath_megamoe.py`, `megamoe_fork/`, `libw6combine.so` |
| Boundary and router | `boundary`, `quant_producer` | One-launch layer boundaries and router; the boundary also writes the next attention input | `fastpath_boundary.py`, `libw4boundary.so`, `fastpath_quant_producer.py`, `lib_seam_quant_cs16.so` |
| Attention | `attn`, `preattn`, `preattn_cand`, `preattn_index`, `preattn_compress` | Decode attention and the chain before it | `fastpath_attn.py`, `w1attn/`, `fastpath_preattn.py`, `w3preattn/` |
| Draft model | `tail_markov_fused`, `tail_dflash_prep`, `tail_ctxkv`, `tail_draft_embed`, `tail_proj_side`, `suffix` | Draft sampling loop, input prep, context projection, and copy drafting for one user | `fastpath_tail_*.py`, `w6_markov.py`, `libw6mk.so`, `fastpath_w9_suffix.py` |
| Step prep and sampling | `tail_swa_meta`, `tail_replay_skip`, `tail_embed`, `topp_fast` | Window metadata, a skipped upload, input embeddings, top-p in one launch | `fastpath_tail_*.py`, `fastpath_topp.py`, `libw6tp.so` |
| Engram, load, memory | `engram_side`, `fastload`, `a12_retire`, `a12_memory_guard` | Engram projection on a side stream; a load filter; scale banks into compressible memory; a free-memory check after capture | `fastpath_engram_side.py`, `fastpath_fastload.py`, `fastpath_a12_retire.py`, `native_scales.py`, `fastpath_a12_memory_guard.py` |

**Compiled kernels.** The recipe has no closed binaries of its own.

| How built | Which | Citation |
|---|---|---|
| JIT at boot, from headers we can edit | The expert kernel: the image's DeepGEMM headers with two replaced by the fork. The launcher checks the image's base header hashes | (GT tree/fastpath/fastpath_megamoe.py:32-37, 56-59) (GT tree/serve/compat.sh:7-11) |
| JIT at boot, Triton in Python | `gt_stage` and Garner McCloud's small glue kernels | (GT tree/fastpath/gracetier_rt/runtime/gracetier_stage.py:147-150) |
| Prebuilt with `nvcc`, source and build script under `tree/kernels/` | `libw4boundary.so`, `lib_seam_quant_cs16.so`, `libw6mk.so`, `libw6tp.so`, `libw6combine.so`, `libw8oproj.so` (its component is not armed here), `w3preattn/libw8topkg.so`, `w1attn/w1_attn_ext.so`, `w3preattn/w3_ext.so`. The two peer3 libraries for the second GPU also ship; their component is not armed. Rebuilds are not byte-identical. We have not rebuilt them | (GT tree/kernels/README.md:20-30) (GT tree/kernels/BUILD-NOTES.md:14-17) |

**gracetier files** (James Meadlock), line counts [S].

| File | Lines | Role |
|---|---|---|
| `fastpath/fastpath_gracetier.py` | 1,091 | The component: load, finalize, wraps, barrier |
| `gracetier_rt/runtime/gracetier_stage.py` | 309 | The staging and counting kernels |
| `gracetier_rt/runtime/gracetier_res.py`, `residency_policy.py` | 701, 389 | Residency manager and decision rule |
| `gracetier_rt/runtime/gracetier_vmm.py` | 466 | VMM layout of the weight tensors |
| `gracetier_rt/runtime/gracetier_pool.py`, `host_tier.py`, `ckpt.py` | 333, 242, 289 | Host pool, registration, checkpoint reads |
| `gracetier_rt/runtime/dg_mega.py`, `dg_moe.py` | 952, 342 | Weight and scale transforms into the expert kernel's layout, and helpers they import |
| Three patches to Garner McCloud's tree: boundary, component list, launcher | 76, 20, 40 | (JM deepseek-v4.1-flash-vllm-gracetier/patches) |

**Our patch sets**, counted from the patch files [S].

| Set | Patch files | Patch lines | Added / removed | Files touched | State |
|---|---|---|---|---|---|
| K1 launch knobs | 3 | 165 | +66 / -8 | 3 | Inert unless variables are set |
| K6 stage skip | 1, plus 3 launcher lines | 489 | +312 / -7 | 3 | Adopted |
| K7 packed 6-bit scale banks | 2, plus 1 launcher line | 1,609 | +1,332 / -29 | 8, two new | Experimental. Frees 3.9 GiB, costs about 3 percent [M] (OURS patches/README.md:9) |

## 7. Limits and open issues

- **Second-GPU assumptions.** Garner McCloud's launcher requires an RTX PRO 6000 and starts a sidecar on it. James Meadlock's launcher patch skips both when `gracetier` is in the list (GT tree/serve/serve.sh:53, 126, 385). `serve.sh doctor` still fails without the card (GT tree/serve/serve.sh:449-455). His measurement script assumes his own box, and his README says a Station without the card was not tested (JM deepseek-v4.1-flash-vllm-gracetier/README.md:62-63). Our port is a directory shim, a stub restore script and a 42-line diff to that script (OURS harness/setup/station_gt_prep.sh:26-37) (OURS harness/setup/card-gt1-station.diff:1-42). It reproduced his one-user numbers: 263.7 and 263.0 against 261.5 and 260.5 [M] (OURS results/logs/gt1-first-boot-card.log:34-35).
- **Utilization.** 0.98 boots. 0.99 runs out of memory at boot [M]. The public recipe ships 0.95 (GT tree/config/recipe.env:11).
- **Context cache.** 1.49M tokens at 1M against 2.71M for the public recipe; 0.52M at 262K [M] (OURS results/summary.json, `kv_tokens`).
- **Batch-dependent output.** With several requests in a batch the unmodified engine does not reproduce its own one-at-a-time output [M]. Cause not verified.
- **No token identity with stock vLLM.** 3 of 40 greedy answers have the same text [M]. The other 37 split at a median of token 27; at the split, stock's top two candidates were within 0.5 log-probability in 22 cases and more than 1.0 apart in 7 (OURS results/identity/stock-vs-engine.txt). Stock vLLM itself gave the same text on only 3 of 40 answers between two boots on this box, the same three prompts [M] (OURS results/identity/stock-vs-stock.txt), so this check cannot separate the engine from stock's own variation between boots. Cause not verified. Garner McCloud states that the stack is not bit-identical to stock vLLM (GT tree/docs/ARCHITECTURE.md:71-74).
- **Inferred contracts.** gracetier assumes one `execute_model` call is one step on one stream, and checks what it can see (GT tree/fastpath/fastpath_gracetier.py:92-97).
- **Evidence.** One machine, two days, most cells run once; the longest run is a 3.5-hour soak (OURS results/logs/gtq21-run.log). The speeds in the tables are settled speeds on prompts that were also used to warm up. A prompt set the server had not seen ran 5 percent under its settled speed on first contact, and 15 percent under on a freshly booted server [M] (OURS results/logs/gtq22-run.log); the engine adapts its resident set to the traffic, and the first timed run of the code prompt is 4 to 13 percent under the session median [M] (OURS results/summary.json, `code_prompt_steps`); on the coding prompt it is 18 to 21 percent under [M] (OURS results/raw).
- **Not separated.** The one-user gain over stock vLLM is part step rate and part draft width: 2.0 times the steps per second and 1.5 times the tokens per step on the code prompt [M]. With the draft count fixed at stock's 3 it is 2.5 times [M] (OURS results/summary.json, `code_prompt_steps`). We have no run of this engine with drafting off.

## 8. Questions for the vLLM and SGLang teams

1. **vLLM, upstreaming.** Would you accept a per-expert host tier in `DeepseekV4MegaMoEExperts`: 384 groups, H resident, a staging call between router and expert kernel? Today it takes wraps of `__init__`, `finalize_weights`, `forward`, the weight-name filter, `load_model` and `execute_model` (GT tree/fastpath/fastpath_gracetier.py:970-1091).
2. **vLLM, step hook.** Could there be a supported pre-step callback on the step's stream? Swaps need that.
3. **vLLM, UVA offload.** How are parameters chosen under `--cpu-offload-gb`? Could it take a per-expert map and a resident scratch, so the expert kernel reads GPU memory? We think it needs both to match this. We could not read the backend here.
4. **SGLang.** Is the SM90 or SM120 gate a correctness limit, or untested on SM100 and SM103? With it removed we scored 195 of 200 on GSM8K [M] (OURS results/raw/sglang-patched/a-quality.md). What blocks speculative decoding and the FlashInfer MXFP4 runner under Paged Experts?
5. **Both.** This design leans on two driver features: one physical allocation mapped behind 40 tensors, and generic-compression allocations for scales. Are both safe to rely on for the GB300?
6. **Measurement.** What would you want measured? We can offer cold experts per layer call by batch size, copy rate by block size, token identity against stock vLLM with components switched one at a time, and a long soak.

## Source roots

| Root | What it is |
|---|---|
| GT | The built engine tree as it ran on the Station before our patches: Garner McCloud's tree at `8043a68` with James Meadlock's lane patches and gracetier applied by the upstream build. `tree/` is the engine, `overlay-vllm/` the vLLM overlay, `policy.env` and `launch-gt1.sh` the launch files. Line numbers are for that built copy. It is produced by the setup steps in this repo and is not redistributed here. Against upstream at `8043a68` it differs in `serve/serve.sh`, `config/recipe.env`, `fastpath/fastpath_boundary.py`, `fastpath/components.py`, `fastpath/w5_markov.py`, `fastpath/w5_markov_gemv.py`, `kernels/w6-moe/markov/w6mk.cu` and the rebuilt `fastpath/libw6mk.so`, and it adds `fastpath/fastpath_gracetier.py`, `fastpath/gracetier_rt/`, `fastpath/gracetier-res.json` and `fastpath/hook/rowmap-allres.json` |
| JM | James Meadlock's recipes, `github.com/J-M-Recipes/recipes` at `486221ebccfc831c56dd979c808948faba652fb9`, directory `recipes/dgx-station-gb300` |
| DEC | gracetier source, `github.com/jmeadlock/gb300-dsv41-decode` at `fadc3d2eb7289189d82ba40cbbdc9b3f10b357a2` |
| TC | Tomas Cupr's repo, `github.com/tomascupr/dgx-station-recipes` at `3f25e5fff74c4428179ce49efb0f7c51822dfb18`. Holds the stock vLLM launch script we measured |
| SGL | SGLang, `main` at `9304abb5321dcf27b28235d53060ecc1438df0f5`. The image we ran was built from `6b374a55`, six commits older |
| OURS | This repo |

Conventions: a citation of a patch file gives lines of the patch file itself. The built tree under GT is before our patches, so the stage-skip kernel and the packed-scale code are cited from the patch files. `rowmap-allres.json` and `gracetier-res.json` are single-line files, so `:1` means the whole file.

Not available to us: the source of vLLM's offload backend (`--offload-backend uva`, `--cpu-offload-gb`, `--cpu-offload-params`). Everything this page says about how stock vLLM offload works comes from the launch script's flags and comments and from James Meadlock's recipe README, and is marked.

66 of the 111 citations were re-opened by script and each contains the quoted text at the cited lines. That shows the text is there, not that our reading of it is right.
