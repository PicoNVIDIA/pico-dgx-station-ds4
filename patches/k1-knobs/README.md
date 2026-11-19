# K1 patches

Against the built engine tree. Apply from the directory that holds `tree/`:

```bash
for p in K1-01-suffix-onereq-draftcount K1-02-quant-producer-expect-widths K1-03-serve-knobs; do
  patch -d tree -p1 --dry-run < patches/k1-knobs/$p.patch && patch -d tree -p1 < patches/k1-knobs/$p.patch
done
```

Checked on a laptop first: `python3 -m py_compile`, `bash -n`, `patch -p1 --dry-run` on a copy, and a local
import of the two Python files with the new variables set. Since booted on the Station: inert by default (the
control boot matched the reference on 40 of 40 prompts), and the knobs were then used for the draft-count runs.

| Patch | File | Adds |
|---|---|---|
| K1-01 | `fastpath/fastpath_w9_suffix.py` | `DS41_K1_FIXED` (1..7), `DS41_K1_LO` (1..6), `DS41_K1_COST` (float or `auto`); a FULL 1 x (1+k) graph for the new width |
| K1-02 | `fastpath/fastpath_quant_producer.py` | `DS41_QP_EXPECT_WIDTHS` (`6,8`, `8`, `6`, `none`); the check stays exact |
| K1-03 | `serve/serve.sh` | Forwards the four variables into the container. `DS41_SPEC_SCHEDULE`, `DS41_CAPTURE_SIZES`, `DS41_BATCHED_TOKENS`, `DS41_LONG_PREFILL` |

All defaults are the shipped values. With nothing set the three patches are meant to change nothing.

Set the variables as extra lines in `policy.env`. Every line must be `NAME=value`: no comments, no blank lines,
no quotes (`launch-gt1.sh` feeds the lines straight to `env`).

Remember: the first schedule entry `[1,hi,k]` sets the 2..hi band. One request is `DS41_K1_FIXED` / `DS41_K1_LO`.
