#!/usr/bin/env python3
"""K6 logic test for the stage-skip patch. No torch, no triton, no GPU: python3 t_stage_skip_logic.py <dir holding tree/>

What runs here is the PATCHED source, not a copy of it:
  * the three kernel bodies of tree/fastpath/gracetier_rt/runtime/gracetier_stage.py (gt_stage_kernel,
    gt_stage_count_kernel, gt_stage_skip_kernel), executed program by program through a small pure-Python stand-in
    for the Triton calls they use, launched by the real gt_stage() launcher;
  * stage(), _skip_cfg(), _skip_finalize(), _skip_on_swap(), _skip_tick(), skip_config(), skip_delta(), lane_of()
    of tree/fastpath/fastpath_gracetier.py, over a small stand-in for torch whose tensors live in one fake address
    space with the VMM aliasing modelled (private hot part + straddle granule, shared scratch per lane).
The residency swap is a transcription of gracetier_res._actuate's five steps (weights and tables), followed by the
on_swap callback exactly where the patch calls it.

Two worlds get the same picks and the same swaps: `ref` (no skip: the first build's kernels, one scratch) and `skp`
(skip on). After every staging the test reads, as MegaMoE would, every picked group of the layer and compares it with
the true bytes of the expert the layer's table names. It also checks that the pick counters the residency policy reads
are identical in both worlds, that copies + skips = the reference's copies, and that no program of a launch touches a
word another program writes.

What this cannot show: that Triton compiles the new kernel, its speed, and anything about real CUDA graphs.
"""
import bisect
import importlib.util
import random
import sys
import types

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
FP = ROOT + "/tree/fastpath"
RNG = random.Random(1234)


# ------------------------------------------------------------------------------------------------ fake memory
class Mem:
    """One address space. Plain allocations are backed 1:1; a reserved VA range is backed through map()."""

    def __init__(self):
        self.d, self.top, self.maps, self.los, self.trace, self.pid = {}, 1 << 20, [], [], None, None

    def alloc(self, nbytes, fill=0, step=8):
        a = self.top
        self.top += (nbytes + 4095) // 4096 * 4096 + 4096
        for o in range(0, nbytes, step):
            self.d[a + o] = fill() if callable(fill) else fill
        return a

    def reserve(self, nbytes):
        a = self.top
        self.top += (nbytes + 4095) // 4096 * 4096 + 4096
        return a

    def map(self, va, nbytes, phys):
        i = bisect.bisect(self.los, va)
        self.los.insert(i, va)
        self.maps.insert(i, (va, va + nbytes, phys))

    def res(self, addr):
        i = bisect.bisect(self.los, addr) - 1
        if i >= 0:
            lo, hi, ph = self.maps[i]
            if lo <= addr < hi:
                return ph + (addr - lo)
        return addr

    def rd(self, addr, kind="r"):
        p = self.res(addr)
        if self.trace is not None:
            self.trace[kind].setdefault(p, set()).add(self.pid)
        return self.d[p]            # KeyError = a read outside every allocation

    def wr(self, addr, v, kind="w"):
        p = self.res(addr)
        if p not in self.d:
            raise KeyError(f"write outside every allocation: {addr:#x}")
        if self.trace is not None:
            self.trace[kind].setdefault(p, set()).add(self.pid)
        self.d[p] = v


MEM = Mem()


# ------------------------------------------------------------------------------------------------ fake torch
class DT:
    def __init__(self, name, size):
        self.name, self.size = name, size


def _numel(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


class FT:
    """A tensor = (address, shape, dtype) over MEM. Only what the patched code calls."""
    is_cuda = True

    def __init__(self, addr, shape, dtype):
        self.addr, self.shape, self.dtype = addr, tuple(int(s) for s in shape), dtype

    def numel(self): return _numel(self.shape)
    def dim(self): return len(self.shape)
    def is_contiguous(self): return True
    def data_ptr(self): return self.addr
    def element_size(self): return self.dtype.size
    def detach(self): return self
    def cpu(self): return self
    def to(self, *a, **k): return self
    def contiguous(self): return self
    def view(self, *s): return FT(self.addr, (self.numel(),), self.dtype)
    reshape = view

    def __getitem__(self, i):
        inner = _numel(self.shape[1:]) * self.dtype.size
        if isinstance(i, slice):
            a, b, _ = i.indices(self.shape[0])
            return FT(self.addr + a * inner, (b - a,) + self.shape[1:], self.dtype)
        return FT(self.addr + int(i) * inner, self.shape[1:], self.dtype)

    def _addrs(self):
        return [self.addr + k * self.dtype.size for k in range(self.numel())]

    def fill_(self, v):
        for a in self._addrs():
            MEM.wr(a, int(v))
        return self

    def add_(self, v):
        for a in self._addrs():
            MEM.wr(a, MEM.rd(a) + int(v))
        return self

    def tolist(self):
        flat = [MEM.rd(a) for a in self._addrs()]
        if len(self.shape) == 2:
            n = self.shape[1]
            return [flat[i * n:(i + 1) * n] for i in range(self.shape[0])]
        return flat if self.shape else flat[0]


class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _mk(shape, dtype, fill):
    t = FT(MEM.alloc(max(1, _numel(shape)) * dtype.size, 0, dtype.size), shape, dtype)
    if fill:
        t.fill_(fill)
    return t


def _tensor(data, dtype=None, device=None):
    shape = []
    d = data
    while isinstance(d, list):
        shape.append(len(d))
        d = d[0] if d else 0
    flat = data
    for _ in shape[1:]:
        flat = [x for row in flat for x in row]
    t = _mk(shape, dtype, 0)
    for a, v in zip(t._addrs(), flat):
        MEM.wr(a, int(v))
    return t


torch = types.ModuleType("torch")
torch.int32, torch.int64, torch.Tensor = DT("int32", 4), DT("int64", 8), FT
torch.tensor = _tensor
torch.zeros = lambda *shape, dtype=None, device=None: _mk(shape[0] if isinstance(shape[0], tuple) else shape, dtype, 0)
torch.full = lambda shape, v, dtype=None, device=None: _mk(shape, dtype, v)
torch.inference_mode = lambda *a: _Ctx()
torch.device = lambda *a: "dev"
torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: False, synchronize=lambda *a: None)


# ------------------------------------------------------------------------------------------------ fake triton
def _a(x):
    return x.a if isinstance(x, V) else x


def _bin(x, y, f):
    a, b = _a(x), _a(y)
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        return [f(p, q) for p, q in zip(a, b)]
    if isinstance(a, list):
        return [f(p, b) for p in a]
    if isinstance(b, list):
        return [f(a, q) for q in b]
    return f(a, b)


class PT:
    def __init__(self, size):
        self.size = size


class V:
    """A Triton value: one int, or a block of ints."""

    def __init__(self, a):
        self.a = a

    def __add__(self, o): return o.__radd__(self) if isinstance(o, Ptr) else V(_bin(self, o, lambda p, q: p + q))
    def __radd__(self, o): return V(_bin(o, self, lambda p, q: p + q))
    def __sub__(self, o): return V(_bin(self, o, lambda p, q: p - q))
    def __rsub__(self, o): return V(_bin(o, self, lambda p, q: p - q))
    def __mul__(self, o): return V(_bin(self, o, lambda p, q: p * q))
    def __rmul__(self, o): return V(_bin(o, self, lambda p, q: p * q))
    def __floordiv__(self, o): return V(_bin(self, o, lambda p, q: p // q))
    def __mod__(self, o): return V(_bin(self, o, lambda p, q: p % q))
    def __and__(self, o): return V(_bin(self, o, lambda p, q: p & q))
    def __eq__(self, o): return V(_bin(self, o, lambda p, q: int(p == q)))
    def __ne__(self, o): return V(_bin(self, o, lambda p, q: int(p != q)))
    def __lt__(self, o): return V(_bin(self, o, lambda p, q: int(p < q)))
    def __gt__(self, o): return V(_bin(self, o, lambda p, q: int(p > q)))
    def __ge__(self, o): return V(_bin(self, o, lambda p, q: int(p >= q)))

    def __bool__(self):
        assert not isinstance(self.a, list), "a block used as a condition"
        return bool(self.a)

    def to(self, dt):
        return Ptr(self.a, dt.size) if isinstance(dt, PT) else self


class Ptr:
    def __init__(self, addr, size):
        self.addr, self.size = addr, size

    def __add__(self, o): return Ptr(_bin(self.addr, o, lambda p, q: p + q * self.size), self.size)
    __radd__ = __add__


PID = [0, 0]
POISON = 0x5EED5EED5EED


def _load(p, mask=None, other=None):
    if isinstance(p.addr, list):
        m = _a(mask) if mask is not None else [1] * len(p.addr)
        return V([MEM.rd(a) if mm else (POISON if other is None else other) for a, mm in zip(p.addr, m)])
    assert mask is None
    return V(MEM.rd(p.addr))


def _store(p, v, mask=None):
    if isinstance(p.addr, list):
        vals = _a(v) if isinstance(_a(v), list) else [_a(v)] * len(p.addr)
        m = _a(mask) if mask is not None else [1] * len(p.addr)
        for a, x, mm in zip(p.addr, vals, m):
            if mm:
                assert x != POISON, "a masked-off load was stored"
                MEM.wr(a, x)
        return
    assert mask is None and not isinstance(_a(v), list)
    MEM.wr(p.addr, _a(v))


def _atomic_add(p, v, mask=None, sem=None):
    addrs = p.addr if isinstance(p.addr, list) else [p.addr]
    vals = _a(v) if isinstance(_a(v), list) else [_a(v)] * len(addrs)
    m = _a(mask) if mask is not None else [1] * len(addrs)
    for a, x, mm in zip(addrs, vals, m):
        if mm:
            MEM.wr(a, MEM.rd(a, "a") + x, "a")


def _mult(p, n):
    assert not isinstance(p.addr, list) and p.addr % n == 0, "pointer not aligned"
    return p


class Jit:
    """triton.jit: kernel[(grid)](*args) runs every program once, in a random order, and checks for races."""
    launches = 0

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        def launch(*args, num_warps=None):
            conv = [Ptr(a.addr, a.dtype.size) if isinstance(a, FT) else a for a in args]
            progs = [(c, q) for c in range(grid[0]) for q in range(grid[1])]
            RNG.shuffle(progs)
            MEM.trace = {"r": {}, "w": {}, "a": {}}
            for c, q in progs:
                PID[0], PID[1] = c, q
                MEM.pid = (c, q)
                self.fn(*conv)
            tr, MEM.trace = MEM.trace, None
            for addr, ws in tr["w"].items():        # a plainly written word: touched by that one program only
                users = ws | tr["r"].get(addr, set()) | tr["a"].get(addr, set())
                assert len(users) == 1, f"race: word {addr:#x} written by {ws}, used by {users}"
            for addr in tr["a"]:                    # an atomic target: never read or written plainly in the launch
                assert addr not in tr["r"] and addr not in tr["w"], f"atomic target {addr:#x} also used plainly"
            Jit.launches += 1
        return launch


tl = types.ModuleType("triton.language")
tl.constexpr, tl.int32, tl.int64 = object, DT("i32", 4), DT("i64", 8)
tl.program_id = lambda i: V(PID[i])
tl.zeros = lambda shape, dtype=None: V([0] * shape[0])
tl.arange = lambda a, b: V(list(range(a, b)))
tl.sum = lambda v, axis=0: V(sum(v.a))
tl.load, tl.store, tl.atomic_add, tl.multiple_of = _load, _store, _atomic_add, _mult
tl.pointer_type = lambda dt: PT(dt.size)
triton = types.ModuleType("triton")
triton.jit, triton.language = Jit, tl
hook = types.ModuleType("fastpath_hook")
hook.activate = lambda *a, **k: None
hook.bounded = lambda *a, **k: _Ctx()
hook.register = lambda *a, **k: (lambda f: f)
sys.modules.update({"torch": torch, "triton": triton, "triton.language": tl, "fastpath_hook": hook})


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GS = _load_mod("k6_gracetier_stage", FP + "/gracetier_rt/runtime/gracetier_stage.py")
FG = _load_mod("fastpath_gracetier", FP + "/fastpath_gracetier.py")
assert GS.HAVE_TRITON and hasattr(GS, "gt_stage_skip_kernel")
FG._S.rt = types.SimpleNamespace(GS=GS)

# small geometry: 8 experts per layer, groups of 38 / 18 words, 3 programs of 4-word blocks, 128-B VMM granules
G, W1, W2, GRAN = 8, 38, 18, 128
B = (W1 * 8, W2 * 8)
GS.NPROG_DEFAULT, GS.BLOCK_WORDS_DEFAULT, GS.SCAN_BLOCK = 3, 4, 8
FG.E, FG.SKIP_TICK, FG.SKIP_LOG_TICKS = G, 8, 4
LOGS = []
FG._log = LOGS.append
COUNT_IDS = 12              # calls with more ids than this take the uncounted kernel (the prefill-sized path)


def up(n, g=GRAN):
    return -(-n // g) * g


def truth(l, e, k):
    r = random.Random(l * 100003 + e * 1009 + k)
    return [r.getrandbits(62) for _ in range((W1, W2)[k])]


class World:
    """One engine: its Grace pool, VMM mappings, tables. skip=None is the first build."""

    def __init__(self, hs, lanes, skip, audit=0, residency=True, hook=True, force=False):
        self.hs, self.hook, self.skip = hs, hook, skip
        self.L = len(hs)
        self.va, self.grace, self.st, self.row, self.g2e = {}, {}, {}, {}, {}
        sh = [max(up(G * B[k]) - min(up(h * B[k]), up(G * B[k])) for h in hs) for k in (0, 1)]
        garbage = lambda: RNG.getrandbits(62)   # noqa: E731
        self.lane_phys = [[MEM.alloc(sh[k], garbage) for k in (0, 1)] for _ in range(lanes)]
        allres = {}
        for l, h in enumerate(hs):
            ex = list(range(G))
            random.Random(l).shuffle(ex)
            row = [0] * G
            for g, e in enumerate(ex):                  # the all-resident map: hot -> 0..h-1, cold -> h..G-1
                row[e] = g
            allres[l] = {"hot": ex[:h], "cold": ex[h:], "row": row}
        if skip:
            FG._S = FG._State()
            FG._S.rt = types.SimpleNamespace(GS=GS)
            FG._S.allres = allres
            FG.STAGE_SKIP_ENV, FG.SCRATCH_LANES_ENV, FG.STAGE_SKIP_AUDIT_ENV = "1", str(lanes), str(audit or "")
            cfg = FG._skip_cfg()
            if force:                                   # negative control only: bypass the boot refusal
                FG._S.skip_cfg = (True, lanes, audit)
            self.cfg = cfg
        for l, h in enumerate(hs):
            C = G - h
            st = FG.GtState(l, allres[l]["hot"], allres[l]["cold"], allres[l]["row"])
            st.lane = FG.lane_of(l, lanes)
            self.row[l], self.g2e[l] = list(allres[l]["row"]), allres[l]["hot"] + allres[l]["cold"]
            vas, gr = [], []
            for k in (0, 1):
                off = min(up(h * B[k]), up(G * B[k]))
                va = MEM.reserve(up(G * B[k]))
                MEM.map(va, off, MEM.alloc(off, garbage))
                MEM.map(va + off, up(G * B[k]) - off, self.lane_phys[st.lane][k])
                for g in range(h):
                    for i, w in enumerate(truth(l, self.g2e[l][g], k)):
                        MEM.wr(va + g * B[k] + 8 * i, w)
                p = MEM.alloc(C * B[k])
                for c in range(C):
                    for i, w in enumerate(truth(l, self.g2e[l][h + c], k)):
                        MEM.wr(p + c * B[k] + 8 * i, w)
                vas.append(va)
                gr.append(p)
            self.va[l], self.grace[l] = vas, gr
            st.src_tab = _tensor([[gr[0] + c * B[0], gr[1] + c * B[1]] for c in range(C)], dtype=torch.int64)
            st.dtab = _tensor(vas, dtype=torch.int64)
            st.cnt = torch.zeros(C, dtype=torch.int32)
            st.b1, st.b2 = B
            if residency:
                st.g2e = _tensor(self.g2e[l], dtype=torch.int32)
                st.ecnt = torch.zeros(G, dtype=torch.int32)
                st.count_ids = COUNT_IDS
            if skip:
                FG._skip_finalize(st, "dev")
            self.st[l] = st
        self.S = FG._S if skip else None
        self.bad = 0
        self.copies_ref = 0

    def group(self, l, g, k):
        return [MEM.rd(self.va[l][k] + g * B[k] + 8 * i) for i in range((W1, W2)[k])]

    def stage(self, l, ids):
        st = self.st[l]
        if self.skip:
            FG._S = self.S
        FG.stage(st, _tensor(ids, dtype=torch.int64))
        for g in sorted({i for i in ids if i >= 0}):          # what MegaMoE reads
            for k in (0, 1):
                self.bad += self.group(l, g, k) != truth(l, self.g2e[l][g], k)
        self.copies_ref += len({i for i in ids if i >= st.H})

    def swap(self, l, g, c):
        """gracetier_res._actuate, steps 1, 2, 3 and 5 (weights and tables), then the on_swap callback."""
        st = self.st[l]
        s = st.H + c
        e_out, e_in = self.g2e[l][g], self.g2e[l][s]
        for k in (0, 1):
            va, gr, n = self.va[l][k], self.grace[l][k], (W1, W2)[k]
            for i in range(n):
                MEM.wr(va + s * B[k] + 8 * i, MEM.rd(va + g * B[k] + 8 * i))
            for i in range(n):
                MEM.wr(va + g * B[k] + 8 * i, MEM.rd(gr + c * B[k] + 8 * i))
            for i in range(n):
                MEM.wr(gr + c * B[k] + 8 * i, MEM.rd(va + s * B[k] + 8 * i))
        self.row[l][e_in], self.row[l][e_out] = g, s
        self.g2e[l][g], self.g2e[l][s] = e_in, e_out
        if st.g2e is not None:
            st.g2e[g].fill_(e_in)
            st.g2e[s].fill_(e_out)
        if self.skip and self.hook:
            FG._S = self.S
            FG._skip_on_swap(l, c)

    def tick(self):
        if self.skip:
            FG._S = self.S
            FG._skip_tick(None)

    def stats(self):
        return [sum(col) for col in zip(*self.S.skip_stat.tolist())]


def run(worlds, steps, swaps, seed, big=0.1):
    """The same picks and swaps into every world. Picks repeat from step to step, as the measured traffic does."""
    rng = random.Random(seed)
    w0 = worlds[0]
    prev = {l: [] for l in range(w0.L)}
    for _ in range(steps):
        if swaps and rng.random() < 0.35:
            for _ in range(rng.randint(1, 2)):
                l = rng.randrange(w0.L)
                g, c = rng.randrange(w0.hs[l]), rng.randrange(G - w0.hs[l])
                for w in worlds:
                    w.swap(l, g, c)
        for w in worlds:
            w.tick()
        for l in range(w0.L):
            n = rng.choice((2, 4, 6)) if rng.random() > big else rng.choice((18, 30))     # > COUNT_IDS: uncounted path
            ex = [rng.choice(prev[l]) if prev[l] and rng.random() < 0.6 else rng.randrange(G) for _ in range(n)]
            prev[l] = ex
            ids = [w0.row[l][e] if rng.random() > 0.05 else -1 for e in ex]
            for w in worlds:
                w.stage(l, ids)


def counters_equal(a, b):
    for l in a.st:
        assert a.st[l].cnt.tolist() == b.st[l].cnt.tolist(), f"cold-slot pick counter differs, layer {l}"
        if a.st[l].ecnt is not None:
            assert a.st[l].ecnt.tolist() == b.st[l].ecnt.tolist(), f"per-expert pick counter differs, layer {l}"


def case(name, hs, lanes, steps=400, swaps=True, audit=0, residency=True, seed=7):
    ref = World(hs, 1, False, residency=residency)
    skp = World(hs, lanes, True, audit=audit, residency=residency)
    run([ref, skp], steps, swaps and residency, seed)
    st = skp.stats()
    assert ref.bad == 0, f"{name}: the reference itself read wrong bytes ({ref.bad})"
    assert skp.bad == 0, f"{name}: skip world read {skp.bad} wrong group(s)"
    counters_equal(ref, skp)
    assert st[0] + st[1] == ref.copies_ref == skp.copies_ref, f"{name}: copies {st[0]} + skips {st[1]} != {ref.copies_ref}"
    assert st[3] == 0 and skp.S.skip_off_why is None, f"{name}: audit mismatch {st[3]}"
    assert (st[2] > 0) == bool(audit), f"{name}: audited {st[2]} with audit {audit}"
    assert st[1] > 0, f"{name}: nothing was skipped"
    tot = skp.S.skip_tot
    assert tot[0] + tot[1] <= st[0] + st[1] and tot[0] + tot[1] > 0, f"{name}: host totals {tot} vs device {st}"
    print(f"ok  {name}: staged {ref.copies_ref}, copied {st[0]}, skipped {st[1]} ({100 * st[1] / ref.copies_ref:.1f} %), "
          f"audited {st[2]}, swap invalidations {skp.S.skip_swaps}")
    return st[1] / ref.copies_ref


# ------------------------------------------------------------------------------------------------ the cases
H6 = [3] * 6
r1 = case("1 lane, swaps, residency on", H6, 1)
r2 = case("2 lanes, swaps", H6, 2)
r3 = case("3 lanes, swaps", H6, 3)
r6 = case("6 lanes (one per layer), swaps", H6, 6)
assert r1 < r2 < r3 < r6, "more lanes must skip more on the same traffic"
case("1 lane, no swaps", H6, 1, swaps=False)
case("2 lanes, residency off (private key table)", H6, 2, residency=False)
case("2 lanes, audit every skip", H6, 2, audit=1)
case("1 lane, audit 1 in 5", H6, 1, audit=5, seed=11)
case("3 lanes, 5 hot of 8, swaps", [5] * 6, 3, seed=3)

# negative control 1: without the swap callback a stale tag must be caught by this test (so the test can see it)
ref, skp = World(H6, 1, False), World(H6, 1, True, hook=False)
run([ref, skp], 600, True, 7)
assert ref.bad == 0 and skp.bad > 0, "no-callback control: the test did not see the stale slot"
print(f"ok  control: no on_swap callback -> {skp.bad} wrong group reads (the callback is needed)")

# negative control 2: different hot counts in one lane, skip forced on -> wrong bytes; and the boot check refuses
HX = [3, 4, 3, 4, 3, 4]
ref, skp = World(HX, 1, False), World(HX, 1, True, force=True)
assert skp.cfg[0] is False and any("DISABLED" in m for m in LOGS), "different H must disable the skip at boot"
run([ref, skp], 600, False, 7)
assert ref.bad == 0 and skp.bad > 0, "different-H control: the test did not see the aliasing"
print(f"ok  control: different hot counts, skip forced -> {skp.bad} wrong group reads; _skip_cfg() disables it at boot")
ref, skp = World(HX, 2, False), World(HX, 2, True)
assert all(not s.skw for s in skp.st.values()) and skp.S.skip_tags is None
skp.skip = False
run([ref, skp], 100, False, 7)
assert ref.bad == 0 and skp.bad == 0
print("ok  different hot counts, skip not forced: no tags are built, every copy is done, lanes still map")

# audit: a word changed behind the kernel's back is found, repaired for the audited slot, and the skip is turned off
skp = World(H6, 6, True, audit=1)
l = 2
c = 1
gid = skp.st[l].H + c
skp.stage(l, [gid, gid])
skp.tick()
skp.stage(l, [gid])
assert skp.stats()[1] == 1 and skp.stats()[3] == 0
a = skp.va[l][0] + gid * B[0] + 8 * 20
MEM.wr(a, MEM.rd(a) ^ 1)
skp.stage(l, [gid])
st = skp.stats()
assert st[3] == 1 and skp.bad == 0, f"audit: mismatch words {st[3]}, wrong reads {skp.bad}"
for _ in range(FG.SKIP_TICK):
    skp.tick()
assert skp.S.skip_off_why and skp.S.skip_ctl.tolist()[0] == 0 and any("ERROR" in m for m in LOGS)
before = skp.stats()
for _ in range(5):
    skp.stage(l, [gid])
after = skp.stats()
assert after[1] == before[1] and after[0] == before[0] + 5 and skp.bad == 0, "skip must be off after a mismatch"
print("ok  audit: 1 changed word found, slot repaired, skip turned off on the device, later picks all copy")

# pure helpers
assert FG.skip_config("", "", "") == (False, 1, 0) and FG.skip_config("0", "", "") == (False, 1, 0)
assert FG.skip_config("1", "4", "1000") == (True, 4, 1000) and FG.skip_config("", "8", "") == (False, 8, 0)
for bad in (("yes", "", ""), ("1", "0", ""), ("1", "x", ""), ("1", "65", ""), ("", "", "5"), ("1", "", "-1"),
            ("1", "2.5", "")):
    try:
        FG.skip_config(*bad)
    except RuntimeError:
        continue
    raise AssertionError(f"skip_config{bad} did not raise")
assert FG.skip_delta([[5, -2147483648, 0, 0]], [[2147483647, 2147483647, 0, 0]], [0, 0, 0, 0]) == [2 ** 31 + 6, 1, 0, 0]
assert [FG.lane_of(i, 4) for i in range(6)] == [0, 1, 2, 3, 0, 1] and FG.lane_of(39, 1) == 0
try:
    FG._S = FG._State()
    FG._S.allres = {0: {"hot": [0], "cold": [1]}}
    FG.STAGE_SKIP_ENV, FG.SCRATCH_LANES_ENV, FG.STAGE_SKIP_AUDIT_ENV = "1", "2", ""
    FG._skip_cfg()
    raise AssertionError("more lanes than layers did not raise")
except RuntimeError:
    pass
print(f"ok  skip_config / skip_delta / lane_of; {Jit.launches} kernel launches, every one race-checked")
print("ALL OK")
