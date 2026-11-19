# Harness

The scripts that set up the box and took the measurements in this repo. They are a record of how it was run, written for one machine. They are not an installer, and nobody has run them from a fresh clone.

## Where the scripts expect things

The scripts use fixed paths on the box. This repo groups the same files by purpose, so to re-run them the files have to be placed where the scripts look:

| On the box | In this repo |
|---|---|
| `/raid/models/DeepSeek-V4.1-Flash-1017` | fetched by `setup/station_prep.sh` |
| `/raid/work/src/{garner,jm-recipes,gb300-dsv41-decode,tomascupr}` | cloned at pinned commits by `setup/station_prep.sh` |
| `/raid/work/run/*.sh`, `*.py` (flat) | `station_cells.sh`, `smoke.sh`, `identity/*.py`, `sglang/*`, `probes/station_k7_*.sh`, `queues/*` |
| `/raid/work/run/cells/` | `cells/*.sh`, and every file in `cells/defs/` |
| `/raid/work/run/patches/K1`, `K6`, `K7` | `../patches/k1-knobs`, `k6-stage-skip`, `k7-packed-scales` |
| `/raid/work/run/patches/K4`, `K5` | `profile/` |
| `/raid/work/run/no-key` | an empty file, created by `setup/station_stock_bench.sh` |
| `/raid/work/gt/card/card-gt1-station.diff` | `setup/card-gt1-station.diff` |
| `/raid/work/ident/gt275-run1.json` | `../results/identity/captures/gt275-run1.json` |
| `/etc/cdi/nvidia.yaml` | created once with the command in `../PINS.md` |

The scripts also assume passwordless `sudo`, `tmux`, Docker with the NVIDIA runtime, and free ports 30006 (this recipe), 8000 (stock vLLM) and 30010 (SGLang). Long jobs run in tmux and end their log with a `DONE` line.

## What is here

| Path | What it does |
|---|---|
| `setup/station_prep.sh` | Directories, the checkpoint download, image pull, upstream clones at pinned commits |
| `setup/station_stock_baseline.sh`, `setup/station_stock_bench.sh` | Boots stock vLLM with its own Station config and runs the benchmark on it |
| `setup/station_pg61_build.sh` | Builds the engine tree. CPU only |
| `setup/station_gt_prep.sh`, `setup/card-gt1-station.diff` | Layout shim for the missing second GPU card and the patch to the upstream launch card |
| `setup/station_gt_boot.sh` | First boot of the one-GPU engine at the published settings |
| `station_cells.sh` | The runner. One boot per "cell": base settings plus one change, warmed, measured, logged |
| `cells/mkhot.sh` | Builds the expert map for a given number of hot experts per layer |
| `cells/k6_*.sh`, `cells/k7_*.sh` | Apply, report on and reverse the patches |
| `cells/defs/` | Every cell that was run: its settings (`.env`), hooks (`.pre.sh`, `.post.sh`) and the queue lists (`q*.list`) |
| `queues/` | The two queue wrappers that gate speed runs on an audit log |
| `smoke.sh` | One real request after a boot. A healthy status code does not prove the server answers |
| `identity/` | The 40-prompt output identity check |
| `sglang/` | The SGLang attempts, the one-line change (`sglang_e2_entry.sh`), and the same benchmark bridged to SGLang with tokens counted at the client |
| `profile/` | The step profiler hook, its driver and table script, and the copy-log patch and analyzer |
| `probes/` | The three probes behind the packed-scale work: the scale survey, the scale-shift arithmetic check, and the GPU unpack probe |
| `t_lsoak.sh`, `t_ho.sh` | Local tests of the runner's long-soak loop and held-out test, with stubbed tools |

## A cell

```
name|profile
```

`cells/defs/<name>.env` holds the extra `policy.env` lines, `<name>.pre.sh` runs before the boot (a new expert map, a patch) and `<name>.post.sh` after the measurements. The profile lists what to measure; the runner's header documents each token. The runner does not stop when a pre-hook fails, so a cell's log has to be checked for the expert count it booted with.

Comments in the scripts and patches cite notes named `research/...`. Those are working notes and are not in this repo.
