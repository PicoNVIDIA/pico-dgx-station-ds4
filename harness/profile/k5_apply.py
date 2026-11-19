#!/usr/bin/env python3
"""Add the K5 nsys/torch-profiler hook to serve.sh by insertion (the K5 patch file conflicts with the K1-03 edits).
Idempotent. The hook does nothing unless DS41_NSYS_WRAP=1 or DS41_TORCH_PROF=1 is set."""
import sys
p = "/raid/work/gt/gracetier/tree/serve/serve.sh"
patch = "/raid/work/run/patches/K5/k5-serve-profile.patch"
s = open(p).read()
if "nsyspfx" in s:
    print("serve.sh: K5 profile hook already present"); sys.exit(0)
block, in_h1 = [], False
for line in open(patch).read().splitlines(keepends=True):
    if line.startswith("@@"):
        if in_h1:
            break
        in_h1 = True
        continue
    if in_h1 and line.startswith("+"):
        block.append(line[1:])
assert len(block) >= 8 and any("nsyspfx=(" in b for b in block), "could not read the hook block from the patch"
lines = s.splitlines(keepends=True)
idx = [i for i, l in enumerate(lines) if l.strip() == '"${OVERLAY_MOUNTS[@]}")']
assert len(idx) == 1, f"mounts anchor found {len(idx)} times"
lines[idx[0] + 1:idx[0] + 1] = block
s = "".join(lines)
for old, new in (('"${keyargs[@]}" "${mounts[@]}" "${envs[@]}" --entrypoint', '"${keyargs[@]}" "${mounts[@]}" "${nsysmnt[@]}" "${envs[@]}" --entrypoint'),
                 ('    vllm serve --model /model', '    "${nsyspfx[@]}" vllm serve --model /model'),
                 ('"${effort[@]}" "${offload[@]}" \\\n', '"${effort[@]}" "${offload[@]}" "${nsysprof[@]}" \\\n')):
    assert s.count(old) == 1, f"anchor found {s.count(old)} times: {old!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print(f"serve.sh: K5 profile hook added ({len(block)} lines + 3 edits)")
