# Pinned versions

Everything the measurements in this repo ran on. The setup scripts check out these commits.

| What | Version | Used for |
|---|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `08c8657f02291e11b6f6b63e0e16014f8e748518` | stock weights, nothing re-quantized |
| Container, engine | `vllm/vllm-openai:v0.31.0`, arm64, `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3` | every boot of the engine in this recipe |
| Container, stock baseline | `vllm/vllm-openai:v0.32.0`, arm64, `sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34` | the "Stock vLLM" column only |
| Engine source | `github.com/garnermccloud/dgx-station-recipes` at `8043a6811a5feda70fdfb6e827eebb2ec25c65d1` (2026-10-28) | the engine and `bench_full.py` |
| Recipe source | `github.com/J-M-Recipes/recipes` at `486221ebccfc831c56dd979c808948faba652fb9` (2026-11-16) | lane patches, the one-GPU launch card |
| Expert tier source | `github.com/jmeadlock/gb300-dsv41-decode` at `fadc3d2eb7289189d82ba40cbbdc9b3f10b357a2` (2026-11-09) | the one-GPU expert tier and its prompt sets |
| Stock baseline script | `github.com/tomascupr/dgx-station-recipes` at `3f25e5fff74c4428179ce49efb0f7c51822dfb18` (2026-11-14) | the stock vLLM launch command |
| SGLang | `lmsysorg/sglang:nightly-dev-cu13-20261119-6b374a55` | the "SGLang (one line patched)" column only |

## Machine

| Part | Value |
|---|---|
| Machine | DGX Station, one GB300 (256,703 MiB reported), 72 Grace cores, 494 GiB of CPU memory |
| OS | DGX OS 7.6.0 (Ubuntu 24.04.4), kernel `6.17.0-1029-nvidia-64k` |
| GPU driver | 610.43.02, CUDA 13.3 |
| Second GPU card | none |

Two things our machine needed before the engine would boot. Both may be specific to how that one machine was set up:

- Its default boot kernel was a 7.0 kernel for which the installed 610 driver had no modules, so `nvidia-smi` could not reach the GPU. Booting `6.17.0-1029-nvidia-64k` fixed it.
- It had no container device spec, and the launcher passes the GPU by device name. One command creates it: `sudo nvidia-ctk cdi generate --device-name-strategy=index --device-name-strategy=uuid --output=/etc/cdi/nvidia.yaml`.
