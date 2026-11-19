# Patches

All three sets apply to the built engine tree (Garner McCloud's engine with James Meadlock's expert tier), from the directory that holds `tree/`. All are off unless their environment lines are set.

| Directory | What it adds | Status |
|---|---|---|
| `k1-knobs/` | Launch settings as environment lines: one-user draft count, draft schedule, capture sizes, prefill chunk size | In the tree for every boot of this recipe's profiles. Inert by default: the control boot matched the reference on 40 of 40 prompts |
| `k6-stage-skip/` | Skips the copy of a cold expert that is still in the layer's scratch slot. Optional extra scratch slots | Adopted. Audit boot: 217,623 skips checked byte for byte, 0 differing words |
| `k7-packed-scales/` | Stores the experts' block scales as 6 bits, not 8, which frees 3.9 GiB of GPU memory | Experimental. Passed four gates on the box, including a byte audit of 4.07 billion output words. Costs about 3% of speed. Not reviewed line by line |

Apply order: `k1-knobs`, then `k6-stage-skip`, then `k7-packed-scales` (which needs the stage skip). The hooks in `harness/cells/` (`k6_apply.sh`, `k7_apply.sh`, `k7_revert.sh`) apply and reverse them on the box and check the result.

Each directory keeps the unit tests of its pure-Python logic. They run on a laptop with CPU PyTorch; none of them compiles or runs a GPU kernel.

The patches are diffs against other people's code. The engine itself is not redistributed here: the setup scripts fetch it from upstream.
