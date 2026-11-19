#!/usr/bin/env python3
"""K7 logic test for the gt_stage part of 0002-gracetier-sf-packed-host.patch. No torch, no triton, no GPU:

    python3 t_k7_stage_logic.py <dir holding the PATCHED tree/> [<k6-stage-skip/t_stage_skip_logic.py>]

It reuses the K6 test's stand-ins (a fake address space with the VMM aliasing, a fake torch, a pure-Python Triton
that runs every program of a launch and checks that no program touches a word another program writes): the part of
k6-stage-skip/t_stage_skip_logic.py above its "the cases" line is executed as it is, against the patched tree, and the
worlds below are built from its World class.

Three worlds get the same picks and swaps: `ref` (the first build's kernels), `skp` (K6 stage skip) and `k7` (K6
stage skip + the K7 scale word: st.skw carries sfp, so the real gt_stage() launches gt_stage_sfp_kernel). Checked:
every group MegaMoE would read holds the right bytes in all three; the k7 world's copies, skips, audits and pick
counters equal the skp world's exactly (the new kernel is the skip kernel plus one store); after every launch the
K7 word holds the address of the layer that was staged; and it is written by program (0, 0) alone.

What this cannot show: that Triton compiles gt_stage_sfp_kernel, and anything about real CUDA graphs.
"""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
K6 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                        "k6-stage-skip", "t_stage_skip_logic.py")
MARK = "# ------------------------------------------------------------------------------------------------ the cases"
src = open(K6).read()
assert src.count(MARK) == 1, "the K6 test changed: its 'the cases' line is not where this test cuts it"
ns = {"__name__": "k6_standin"}
sys.argv = [K6, ROOT]
exec(compile(src.split(MARK)[0], K6, "exec"), ns)
World, run, counters_equal, FG, GS, MEM, torch, Jit = (ns[k] for k in ("World", "run", "counters_equal", "FG", "GS",
                                                                      "MEM", "torch", "Jit"))
assert hasattr(GS, "gt_stage_sfp_kernel"), "the patched gracetier_stage.py has no gt_stage_sfp_kernel"
BASE_ADDR = 0x7F0000000000


class K7World(World):
    """The K6 skip world whose layers also pass the K7 scale word to gt_stage, as _sfp_finalize arranges it."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.word = torch.zeros(1, dtype=torch.int64)
        self.addr = {l: BASE_ADDR + l * 256 for l in self.st}
        for l, st in self.st.items():
            assert "skip" in st.skw
            st.skw = dict(st.skw, sfp=(self.word, self.addr[l]))
        self.word_bad = self.stages = 0

    def stage(self, l, ids):
        MEM.wr(self.word.addr, 0)                       # what the audit's dense re-run leaves behind
        super().stage(l, ids)
        self.stages += 1
        self.word_bad += MEM.rd(self.word.addr) != self.addr[l]


def case(name, hs, lanes, steps=300, swaps=True, audit=0, residency=True, seed=7):
    ref = World(hs, 1, False, residency=residency)
    skp = World(hs, lanes, True, audit=audit, residency=residency)
    k7 = K7World(hs, lanes, True, audit=audit, residency=residency)
    run([ref, skp, k7], steps, swaps and residency, seed)
    a, b = skp.stats(), k7.stats()
    assert ref.bad == 0 and skp.bad == 0 and k7.bad == 0, f"{name}: wrong group reads {ref.bad} / {skp.bad} / {k7.bad}"
    counters_equal(ref, k7)
    assert a == b, f"{name}: the K7 kernel's copy / skip / audit counters {b} differ from the skip kernel's {a}"
    assert a[1] > 0 and (a[2] > 0) == bool(audit), f"{name}: skips {a[1]}, audited {a[2]}"
    assert k7.word_bad == 0 and k7.stages > 0, f"{name}: the K7 word was wrong after {k7.word_bad} of {k7.stages} launches"
    print(f"ok  {name}: {k7.stages} launches, the K7 word named the staged layer after each; copied {b[0]}, skipped "
          f"{b[1]}, audited {b[2]} = the skip kernel's counters; pick counters = the first build's")


H6 = [3] * 6
case("1 lane, swaps, residency on", H6, 1)
case("3 lanes, swaps", H6, 3)
case("2 lanes, residency off (no per-expert counter)", H6, 2, residency=False)
case("2 lanes, audit every skip", H6, 2, audit=1)
case("1 lane, 5 hot of 8, audit 1 in 5", [5] * 6, 1, audit=5, seed=11)

# a launch with no pick at all (prewarm; a call whose picks are all hot) still stores the word
w = K7World(H6, 1, True)
w.stage(2, [-1, -1])
w.stage(4, [0, 1, 2])
assert w.word_bad == 0 and w.stats()[0] == 0, "a launch without cold picks must still store the word and copy nothing"
print("ok  a launch with no cold pick stores the word and copies nothing")

# the word without the skip tables has no kernel to ride in: the launcher refuses
ref = World(H6, 1, False)
st = ref.st[0]
st.skw = {"sfp": (torch.zeros(1, dtype=torch.int64), BASE_ADDR)}
try:
    FG.stage(st, ns["_tensor"]([0, 1], dtype=torch.int64))
    raise AssertionError("sfp without skip did not raise")
except ValueError:
    pass
# a misaligned or zero table address, or a word that is not int64, is refused
k7 = K7World(H6, 1, True)
for bad in ((k7.word, 0), (k7.word, BASE_ADDR + 8), (torch.zeros(1, dtype=torch.int32), BASE_ADDR)):
    st = k7.st[1]
    st.skw = dict(st.skw, sfp=bad)
    try:
        FG.stage(st, ns["_tensor"]([0, 1], dtype=torch.int64))
        raise AssertionError(f"sfp {bad[1]:#x} did not raise")
    except ValueError:
        pass
print(f"ok  sfp without skip, a zero / misaligned address and a non-int64 word are refused; {Jit.launches} kernel "
      f"launches, every one race-checked")
print("ALL OK")
