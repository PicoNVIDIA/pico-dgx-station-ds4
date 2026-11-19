#!/usr/bin/env python3
"""K7 host-side test for 0002-gracetier-sf-packed-host.patch and 0003-megamoe-sf-packed-kernel.patch.
CPU torch + numpy, no triton, no GPU:

    env -u PYTHONPATH uv run --no-project --with torch --with numpy python t_k7_host.py <dir holding the PATCHED tree/> \
        [<dir holding the UNPATCHED (K1 + K6) tree/>]

What runs here is the patched source: runtime/gracetier_sfpack.py (formats, bank, tensor-map arguments, the dense
address range against the real gracetier_vmm.VmmArena with a recording driver), fastpath_gracetier's K7 functions
over stand-in modules, the real runtime/gracetier_res.GtResManager swapping packed rows, fastpath_a12_retire's row
and fastpath_a12_memory_guard's check, megamoe_sfpack's edit list on the real fork header, and fastpath_megamoe's
include-root naming with the variable unset and set. With the second argument the off-path values are compared with
the unpatched tree's.

What this cannot show: anything on a GPU; that the rendered header compiles; what the hardware's 4- / 6-bit tensor
map and decompressing copy really write (experiment E2); cuTensorMapEncodeTiled's verdict on the arguments.
"""
import contextlib
import importlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
BASE = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
FP = os.path.join(ROOT, "tree", "fastpath")
sys.path.insert(0, os.path.join(FP, "gracetier_rt"))
sys.path.insert(0, FP)
for k in list(os.environ):
    if k.startswith("GRACETIER_"):
        del os.environ[k]

FAILS = []
GEN = torch.Generator().manual_seed(7)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def codes(shape, lo=117, hi=127):
    return torch.randint(lo, hi + 1, shape, generator=GEN, dtype=torch.uint8)


def mn_bank(g, n, q, lo=117, hi=127):
    """An MN-major int32 (g, n, q) scale tensor of random codes."""
    t = torch.empty_strided((g, n, q), (n * q, 1, n), dtype=torch.int32)
    t.transpose(1, 2).contiguous()
    v = t.transpose(1, 2)
    v.view(torch.uint8).copy_(codes((g, q, 4 * n), lo, hi))
    return t


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


# ================================================================================================ H. off by default
before = set(sys.modules)
FG = importlib.import_module("fastpath_gracetier")
MM = importlib.import_module("fastpath_megamoe")
new = set(sys.modules) - before
check("H unset: importing the two components imports no K7 module", not any(
    m in new for m in ("megamoe_sfpack", "runtime.gracetier_sfpack", "native_scales", "runtime")), str(sorted(
        m for m in new if "sfpack" in m or "native" in m)))
check("H unset: sfpack_config() is off and megamoe has no variant", FG.sfpack_config() == ("", None, None, ())
      and MM.K7 is None and FG._State().sfp_word is None and FG._State().sfp == {})
check("H 0 means off; probe is phase 0", FG.sfpack_config("0") == ("", None, None, ())
      and FG.sfpack_config("probe") == ("probe", None, None, ()))
check("H a2b6 / a2b4 / kernel-b6 parse; order and audit parse", FG.sfpack_config("a2b6") == ("a2", "b6", "lo", ())
      and FG.sfpack_config("a2b4", "hi", "39,0,19") == ("a2", "b4", "hi", (0, 19, 39))
      and FG.sfpack_config("kernel-b6") == ("kernel", "b6", "lo", ()))
bad_cfgs = (("a2",), ("A2B6",), ("a2b6 ",), ("1",), ("a2b6", "low"), ("a2b6", "", "0,x"), ("", "hi"), ("", "", "3"),
            ("probe", "", "3"), ("kernel-b6", "", "1"), ("kernel-b6", "hi"),
            ("a2b6", "", ",".join(str(i) for i in range(9))))
check("H a typo raises (a bare a2, wrong case, an order or audit without a2, 9 audit layers)",
      all(raises(lambda c=c: FG.sfpack_config(*c), RuntimeError) for c in bad_cfgs))
SRC = open(os.path.join(FP, "fastpath_gracetier.py")).read().split("\n")
K7_LO = next(i for i, l in enumerate(SRC) if l.startswith("# ====") and "K7 packed scales" in l)
K7_HI = next(i for i, l in enumerate(SRC) if l.startswith("def _finalize_layer("))
K7_CALLS = ("_sfp_banks(", "_sfp_hot_gone(", "_sfp_finalize(", "_sfp_ready(", "_sfp_tick(", "_sfpack_probe(",
            "_sfp_rt(", "_sfp_dummy(", "_sfp_storage(", "_sfp_audit_install(")
GUARDS = ('_sfp()[0] == "a2"', '_sfp()[0] in ("a2", "kernel")', '_sfp()[0] == "probe"', "_S.sfp_word is not None")
unguarded, sites = [], 0
for i, line in enumerate(SRC):
    if K7_LO <= i < K7_HI or line.lstrip().startswith(("def ", "#")) or not any(c in line for c in K7_CALLS):
        continue
    sites += 1
    ind, j, ok_ = len(line) - len(line.lstrip()), i, False
    while j > 0 and not ok_:
        j -= 1
        up = SRC[j]
        if not up.strip() or len(up) - len(up.lstrip()) >= ind:
            continue
        ind = len(up) - len(up.lstrip())
        if up.lstrip().startswith(("if ", "elif ")):
            ok_ = any(g in up for g in GUARDS)
            break
        if up.lstrip().startswith("def "):
            break
    if not ok_:
        unguarded.append(i + 1)
check(f"H every call into the K7 code from the load path, the step hook and load end ({sites} sites) sits under a "
      f"test of the mode or of the armed word", sites >= 7 and not unguarded, str(unguarded))
if BASE:
    BMM = load("base_megamoe", os.path.join(BASE, "tree", "fastpath", "fastpath_megamoe.py"))
    check("H unset: the JIT include root has the name it had before the patch", MM.ROOT_NAME == BMM.ROOT_NAME
          and MM.ROOT_NAME.startswith("shared-early-") and "k7" not in MM.ROOT_NAME, MM.ROOT_NAME)
    same = all(open(os.path.join(FP, r), "rb").read() == open(os.path.join(BASE, "tree", "fastpath", r), "rb").read()
               for r in ("megamoe_fork/" + f for f in MM.FILES))
    check("H the fork headers on disk are untouched by the patches", same)
    vend = tempfile.mkdtemp()
    os.makedirs(os.path.join(vend, "include", "deep_gemm", "impls"))
    os.makedirs(os.path.join(vend, "include", "deep_gemm", "scheduler"))
    open(os.path.join(vend, "include", "deep_gemm", "other.cuh"), "w").write("// vendored\n")
    MM.ROOT_BASE = os.path.join(vend, "roots")
    root = MM.build_root(vend)
    same = all(open(os.path.join(root, r), "rb").read() == open(os.path.join(MM.PATCH_DIR, r), "rb").read()
               for r in MM.FILES)
    check("H unset: build_root copies the two fork headers byte for byte", same and os.path.basename(root)
          == BMM.ROOT_NAME)
else:
    print("SKIP H comparisons with the unpatched tree (no second argument)")

# ================================================================================================ A. the format
from runtime import gracetier_sfpack as SP  # noqa: E402
from runtime import gracetier_vmm as GV  # noqa: E402
from runtime import gracetier_res as GR  # noqa: E402
import megamoe_sfpack as MS  # noqa: E402

for fmt in ("b4", "b6"):
    f = SP.FORMATS[fmt]
    for order in ("lo", "hi"):
        d = codes((3, 5, 64), f.lo, f.hi)
        p = SP.pack_codes(d, fmt, order)
        check(f"A {fmt}/{order}: 16 codes -> {f.group_bytes} B, unpack(pack) == codes", p.shape[-1] == 64 // 16
              * f.group_bytes and torch.equal(SP.unpack_codes(p, fmt, order), d))
    allv = torch.arange(f.lo, f.hi + 1, dtype=torch.uint8).repeat(16)[:16 * ((f.hi - f.lo + 16) // 16)]
    check(f"A {fmt}: every packable value survives", torch.equal(SP.unpack_codes(SP.pack_codes(allv, fmt), fmt), allv))
    lo_p, hi_p = SP.pack_codes(d, fmt, "lo"), SP.pack_codes(d, fmt, "hi")
    check(f"A {fmt}: the two orders are different bytes", not torch.equal(lo_p, hi_p))
    bad = d.clone()
    bad[0, 0, 3], bad[2, 4, 60] = f.lo - 1, 128
    check(f"A {fmt}: codes outside {f.lo}..{f.hi} are counted and do not survive", int(SP.outside(bad, fmt).sum())
          == 2 and int((SP.unpack_codes(SP.pack_codes(bad, fmt), fmt) != bad).sum()) == 2)
d = codes((2, 3, 32), 112, 127)
check("A b4/lo is the phase 0 probe's pack (0001)", torch.equal(SP.pack_codes(d, "b4", "lo"), SP.pack_bytes(d)))
x = int.from_bytes(bytes(SP.pack_codes(torch.tensor([64 + i for i in range(16)], dtype=torch.uint8), "b6").tolist()),
                   "little")
check("A b6/lo: code i of a group sits in bits 6i..6i+5 of the 96-bit little-endian group",
      [(x >> (6 * i)) & 63 for i in range(16)] == list(range(16)))
x = int.from_bytes(bytes(SP.pack_codes(torch.tensor([112 + i for i in range(16)], dtype=torch.uint8), "b4").tolist()),
                   "little")
check("A b4/lo: code i of a group sits in bits 4i..4i+3 (the PTX manual's .b4x16_p64 figure: LSB first)",
      [(x >> (4 * i)) & 15 for i in range(16)] == list(range(16)))
d = codes((4, 64))
k6 = SP.kernel_bytes(SP.pack_codes(d, "b6"), "b6")
k4 = SP.kernel_bytes(SP.pack_codes(d, "b4"), "b4")
check("A manual model, b6: tensor memory = code - 64, so a + 64 and b - 64 keep every exponent sum",
      torch.equal(k6.to(torch.int32) + 64, d.to(torch.int32)))
check("A manual model, b4: tensor memory = (code - 112) << 2, NOT code - 112: no constant bias can repair it",
      torch.equal(k4.to(torch.int32), (d.to(torch.int32) - 112) * 4)
      and len(set((d.to(torch.int32) - k4.to(torch.int32)).flatten().tolist())) > 1)
check("A if the hardware zero-extended a nibble (shift 0), b4 would be exact with bias 112",
      torch.equal(SP.kernel_bytes(SP.pack_codes(d, "b4"), "b4", shift=0).to(torch.int32) + 112, d.to(torch.int32)))
for bias in (64, 112):
    ok = True
    for b in range(256):
        for pos in range(4):
            w = (0x7F7F7F7F & ~(0xFF << (8 * pos))) | (b << (8 * pos))
            out, over = SP.bias_word(w, bias)
            want_over = b > 254 - bias
            ok &= over == want_over
            if not want_over:
                ok &= out == sum(((0x7F if i != pos else b) + bias) << (8 * i) for i in range(4))
    check(f"A the kernel's biased-word test, bias {bias}: flags exactly the codes above {254 - bias}, all 4 bytes", ok)
check("A megamoe_sfpack's format table equals gracetier_sfpack's", all(
    MS.FORMATS[k]["bits"] == v.bits and MS.FORMATS[k]["bias"] == v.bias and MS.FORMATS[k]["group"] == v.group_bytes
    and MS.FORMATS[k]["cp"] == v.cp_fmt and MS.config("a2" + k)["tx"] == SP.tx_bytes(k) for k, v in SP.FORMATS.items())
    and SP.tx_bytes("b4") == 256 and SP.tx_bytes("b6") == 384)

# real shapes: one expert's L1 / L2 scales
for fmt, want in (("b4", (368640, 184320)), ("b6", (552960, 276480))):
    got = (SP.packed_bytes(1, 4608, 40, fmt), SP.packed_bytes(1, 5120, 18, fmt))
    check(f"A {fmt}: one expert 737,280 + 368,640 B of dense scales -> {want[0]:,} + {want[1]:,} B", got == want)

# ================================================================================================ B. tensor map
for fmt in ("b4", "b6"):
    for n, q, G in ((4608, 40, 384), (5120, 18, 384)):
        P = SP.packed_col_bytes(n, fmt)
        raw = torch.zeros(2 * q * P, dtype=torch.uint8)           # 2 groups stand for 384: the arguments scale
        pk = SP.PackedBank(raw, 2, n, q, fmt)
        a = SP.tma_args(pk)
        piece = 8 * SP.FORMATS[fmt].group_bytes
        ok = (a["rank"] == 3 and a["box"] == (128, 4, 1) and a["dims"] == (128, n // 32, 2 * q)
              and a["strides"] == (piece, P) and all(s % 32 == 0 for s in a["strides"]) and a["address"] % 32 == 0
              and a["dims"][1] * piece == P and a["dtype"] == SP.FORMATS[fmt].tma_type
              and a["box"][1] * piece == SP.tx_bytes(fmt))
        check(f"B {fmt} n {n} q {q}: 3-D map, box (128 elements, 4 pieces, 1 column), strides {a['strides']} B", ok,
              str(a["dims"]))
seen = []


def fake_encode(args):
    seen.append(args)
    return bytes([len(seen)]) * 128


check("B encode_tma returns the encoder's 128 bytes and refuses another size", SP.encode_tma(a, fake_encode)
      == b"\x01" * 128 and raises(lambda: SP.encode_tma(a, lambda x: b"x" * 64), RuntimeError))
fake_cu = types.SimpleNamespace(CUtensorMapDataType=type("E", (), {"CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B": 15}))
check("B enum values are read by name; a missing name raises and lists what exists", SP._enum(
    fake_cu, "CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B") == 15 and raises(lambda: SP._enum(fake_cu, "NOPE"), RuntimeError))

# ================================================================================================ C. dense address
class FakeDrv:
    def __init__(self):
        self.ops, self.va, self.h = [], 1 << 40, 100

    def granularity(self, device, recommended=True):
        return 2 << 20

    def create(self, n, device):
        self.h += 1
        self.ops.append(("create", n))
        return self.h

    def reserve(self, n, align=0):
        va, self.va = self.va, self.va + n + (64 << 20)
        self.ops.append(("reserve", n))
        return va

    def map(self, va, n, off, h):
        self.ops.append(("map", va, n, off, h))

    def set_access(self, va, n, device):
        pass


drv = FakeDrv()
arena = GV.VmmArena(0, driver=drv)
views = []


def fake_view(owner, ptr, nbytes, dtype, shape, dev):
    views.append((ptr, nbytes, dtype, tuple(shape)))
    n = 1
    for s in shape:
        n *= s
    return torch.zeros(n, dtype=dtype).view(shape)


check("C the K7 word sits in the fork's 512-B control tail: fold control + 384, clear of C11 (+ 256, 32 B)",
      SP.K7_OFFSET == MM.P3FOLD_CTL_OFFSET + 384 and MM.C11_OFFSET + 32 <= SP.K7_OFFSET
      and SP.K7_OFFSET + 8 + 4 * SP.K7_WORDS <= MM.P3FOLD_CTL_OFFSET + 512)
SHAPES = ((4608, 40), (5120, 18))
d1, d2 = SP.dummy_dense(arena, GV, 384, SHAPES, "cpu", view=fake_view)
maps = [o for o in drv.ops if o[0] == "map"]
creates = [o for o in drv.ops if o[0] == "create"]
res = [o for o in drv.ops if o[0] == "reserve"]
check("C the dense-shaped pair: ONE 2 MiB allocation, two reservations of 270 and 136 MiB, 135 + 68 mappings of it",
      creates == [("create", 2 << 20)] and [r[1] for r in res] == [135 * (2 << 20), 68 * (2 << 20)]
      and len(maps) == 203 and all(m[2] == 2 << 20 and m[3] == 0 and m[4] == drv.h for m in maps))
check("C the tensors are MN-major (384, n, q) int32, as fastpath_a12_retire.source_spec and DeepGEMM's layout check "
      "want", all(tuple(t.shape) == (384, n, q) and tuple(t.stride()) == (n * q, 1, n) and t.dtype == torch.int32
                  for t, (n, q) in zip((d1, d2), SHAPES)))
check("C every mapping lies inside a reservation and the poison fill goes through the first one",
      views[0][2] == torch.uint8 and views[0][1] == 2 << 20 and len(arena.maps) == 203)

# ================================================================================================ D. the component
torch.cuda.is_current_stream_capturing = lambda: False
E_, D_, I_, H_ = 12, 512, 256, 7
C_ = E_ - H_
FG.E, FG.HIDDEN, FG.INTER = E_, D_, I_
N1, Q1, N2, Q2 = 2 * I_, D_ // 128, D_, I_ // 128
LOGS = []
FG._log = LOGS.append


class Own:
    def __init__(self, n):
        self.nbytes, self.size, self.granted = n, -(-n // 4096) * 4096, 1


def fake_storage(nbytes, dev):
    return torch.zeros(int(nbytes), dtype=torch.uint8), Own(int(nbytes))


def fresh(fmt="b6", order="lo", audit=()):
    FG._S = FG._State()
    FG._S.allres = {l: {"hot": list(range(H_)), "cold": list(range(H_, E_))} for l in range(3)}
    FG._S.rt = types.SimpleNamespace(GS=types.SimpleNamespace(gt_stage_sfp_kernel=object()), GV=GV, GR=GR)
    FG._S.sfp_cfg = ("a2", fmt, order, tuple(audit))
    dum = tuple(torch.full((1,), -1, dtype=torch.int32).expand(E_ * n * q).contiguous().view(E_, q, n).transpose(1, 2)
                for n, q in ((N1, Q1), (N2, Q2)))
    FG._sfp_dummy = lambda dev: dum
    FG._sfp_storage = fake_storage
    sym = types.SimpleNamespace(buffer=torch.zeros(SP.K7_OFFSET + 64, dtype=torch.int8))
    FG._symm_for = lambda m: sym
    SP.cuda_encode = fake_encode
    return dum, sym


class Mod:
    pass


def finalize(idx, hot_lo=117, cold_hi=127):
    """What _finalize_layer does around the K7 calls, for one stand-in layer."""
    m, st = Mod(), FG.GtState(idx, list(range(H_)), list(range(H_, E_)), list(range(E_)))
    sf1, sf2 = mn_bank(H_, N1, Q1, hot_lo), mn_bank(H_, N2, Q2, hot_lo)
    c1, c2 = mn_bank(C_, N1, Q1, 117, cold_hi).contiguous(), mn_bank(C_, N2, Q2, 117, cold_hi).contiguous()
    truth = (torch.cat([SP.col_bytes(sf1), SP.col_bytes(c1)]).clone(), torch.cat([SP.col_bytes(sf2),
                                                                                  SP.col_bytes(c2)]).clone())
    w1 = torch.zeros(E_, 64, dtype=torch.int8)
    b1, b2, q1, q2 = FG._sfp_banks(m, st, torch.device("cpu"), sf1, c1, sf2, c2)
    par = (torch.equal(SP.col_bytes(q1), truth[0][:1]), torch.equal(SP.col_bytes(q2), truth[1][:1]))
    m._transformed_l1_weights, m._transformed_l2_weights = (w1, b1), (w1, b2)
    del sf1, sf2, q1, q2
    FG._sfp_hot_gone(idx)
    st.skw = {"skip": ("tag", "g2e", idx * E_, "ctl", "stat", 0)}
    FG._sfp_finalize(m, st, torch.device("cpu"))
    m._mp = st
    return m, st, truth, par


for fmt in ("b6", "b4"):
    for order in ("lo", "hi"):
        dum, sym = fresh(fmt, order, audit=(1,))
        lo = SP.FORMATS[fmt].lo
        mods = {l: finalize(l, hot_lo=max(117, lo)) for l in range(3)}
        ok = True
        for l, (m, st, truth, par) in mods.items():
            info = FG._S.sfp[l]
            ok &= all(par) and m._transformed_l1_weights[1] is dum[0] and m._transformed_l2_weights[1] is dum[1]
            for k, pk in enumerate(info["banks"]):
                ok &= pk.mismatch(0, truth[k]) == 0 and pk.nbytes * 8 == pk.dense_nbytes * SP.FORMATS[fmt].bits
                ok &= torch.equal(SP.col_bytes(pk.dense(0, E_)), truth[k])
            ok &= info["hot_gone"] == [True, True] and "handles" not in info
            ok &= (info["dense"] is not None) == (l == 1)
            if l == 1:
                ok &= all(torch.equal(SP.bank_bytes(b), truth[k]) for k, b in enumerate(info["dense"]))
            w, addr = st.skw["sfp"]
            ok &= w is FG._S.sfp_word and addr == FG._S.sfp_desc.data_ptr() + l * 256 and "skip" in st.skw
        check(f"D {fmt}/{order}: 3 layers finalize packed: banks == sources, host side gets the shared dense-shaped "
              f"pair, parity sees group 0, vLLM's hot scales released, audit layer keeps its dense twins", ok)
desc, ctl = SP.k7_views(sym.buffer)
check("D the K7 word: magic armed once, desc word still 0 (gt_stage writes it per call), error word 0",
      ctl.tolist() == [SP.K7_MAGIC, 0, 0, 0] and int(desc[0]) == 0
      and desc.data_ptr() == sym.buffer.data_ptr() + 21009664)
t = FG._S.sfp_desc
check("D the tensor-map table: uint8 [layers, 2, 128], one encoded map per packed bank, 64-B aligned",
      tuple(t.shape) == (3, 2, 128) and t.data_ptr() % 64 == 0 and len({int(t[l, k, 0]) for l in range(3)
                                                                         for k in range(2)}) == 6)
dum, sym = fresh("b4")
check("D a code the format cannot hold refuses the boot (b4: a hot code 111)", raises(
    lambda: finalize(0, hot_lo=111), RuntimeError))
dum, sym = fresh("b6")
st = FG.GtState(0, list(range(H_)), list(range(H_, E_)), list(range(E_)))
FG._S.sfp[0] = {"banks": ()}
check("D without the skip tables the scale word has no launch to ride in: refused", raises(
    lambda: FG._sfp_finalize(Mod(), st, torch.device("cpu")), RuntimeError))
sym.buffer[SP.K7_OFFSET + 12] = 1
st.skw = {"skip": ()}
check("D a SymmBuffer whose K7 word is not zero is refused", raises(
    lambda: FG._sfp_finalize(Mod(), st, torch.device("cpu")), RuntimeError))
sym.buffer[SP.K7_OFFSET + 12] = 0
FG._S.symm_desc = {"how": "vLLM's own (already G = 384)"}
check("D a SymmBuffer that vLLM hands to other layers too is refused (a stale word would reach them)", raises(
    lambda: FG._sfp_finalize(Mod(), st, torch.device("cpu")), RuntimeError))

# _sfp(): the megamoe variant must match, at hook time
FG._S = FG._State()
FG._S.allres = {0: {"hot": [0], "cold": [1]}}
FG.SF_PACK_ENV = "a2b6"
MM.K7 = None
check("D _sfp(): a2b6 without the kernel variant in fastpath_megamoe is refused at once", raises(FG._sfp, RuntimeError))
MM.K7 = MS.config("a2b4")
check("D _sfp(): a kernel variant of the other format is refused", raises(FG._sfp, RuntimeError))
MM.K7 = MS.config("a2b6")
FG.SF_PACK_AUDIT_ENV = "5"
check("D _sfp(): an audit layer the rowmap does not have is refused", raises(FG._sfp, RuntimeError))
FG.SF_PACK_AUDIT_ENV = "0"
check("D _sfp(): accepted when the variant matches", FG._sfp() == ("a2", "b6", "lo", ()))
FG.SF_PACK_ENV = FG.SF_PACK_AUDIT_ENV = ""

# ------------------------------------------------ the audit wrapper, the tick, the ready check
dum, sym = fresh("b6", "lo", audit=(1,))
mods = {l: finalize(l) for l in range(3)}
for l, (m, st, truth, par) in mods.items():       # each layer its own weights tensor: the audit keys on its address
    m._transformed_l1_weights = (torch.zeros(E_, 64, dtype=torch.int8), dum[0])
    FG._S.sfp[l]["w1_ptr"] = int(m._transformed_l1_weights[0].data_ptr())
CALLS = []
SEAM = [False]          # the router asked for the combine in the seam: the kernel then does not write y, and clears it


def fake_mega(y, l1w, l2w, symm, **kw):
    """A stand-in kernel: y = a function of the scale bytes it is given. K7 word != 0: the packed banks of the layer
    whose table address the word holds; 0: the dense tensors passed in. Like the fork (C11 skip_combine), a call
    that finds SEAM set leaves y unwritten and clears the flag."""
    word = int(FG._S.sfp_word[0])
    if word:
        l = (word - FG._S.sfp_desc.data_ptr()) // 256
        cols = [SP.col_bytes(pk.dense(0, E_)) for pk in FG._S.sfp[l]["banks"]]
    else:
        cols = [SP.col_bytes(l1w[1]), SP.col_bytes(l2w[1])]
    CALLS.append(("packed" if word else "dense") + ("-seam" if SEAM[0] else ""))
    if SEAM[0]:
        SEAM[0] = False
        y.copy_(torch.rand(y.shape).to(torch.bfloat16))
        return
    v = float((int(cols[0].to(torch.int64).sum()) + 3 * int(cols[1].to(torch.int64).sum())) % 251)
    y.copy_((torch.arange(y.numel(), dtype=torch.float32).view(y.shape) + v).to(torch.bfloat16))


dg = types.SimpleNamespace(fp8_fp4_mega_moe=fake_mega)
sys.modules["vllm.utils.deep_gemm"] = types.SimpleNamespace(_import_deep_gemm=lambda: dg)
MM.ACTIVE, MM.K7 = True, MS.config("a2b6")
_, err = quiet(FG._sfp_ready, {0, 1, 2})
armed = [m for m in LOGS if "ARMED" in m][-1]
print("     log: " + armed[:330])
check("D _sfp_ready: ARMED line, the audit wrapper installed on DeepGEMM's module attribute",
      "6 bits a code (bias 64" in armed and "released in 3/3" in armed and dg.fp8_fp4_mega_moe is not fake_mega
      and FG._S.sfp_orig is fake_mega and len(FG._S.sfp_audit) == 1)


def call(l, T=6, seam=False):
    m = mods[l][0]
    FG._S.sfp_word.fill_(mods[l][1].skw["sfp"][1])           # what gt_stage's program (0, 0) stores
    SEAM[0] = seam
    y = torch.empty(T, 16, dtype=torch.bfloat16)
    dg.fp8_fp4_mega_moe(y, m._transformed_l1_weights, m._transformed_l2_weights, sym, fast_math=True)
    return y


AUD4 = ["dense", "dense", "packed"]
for l in (0, 1, 2):
    call(l)
call(1, seam=True)
call(1, T=20)
stat = FG._S.sfp_stat.tolist()
check("D audit: an audit layer's call is followed by dense, dense, packed re-runs, other layers pass through; a "
      "call whose combine went to the seam (y unwritten) still audits clean; 0 differing words, control 0",
      CALLS == ["packed", "packed"] + AUD4 + ["packed", "packed-seam"] + AUD4 + ["packed"] + AUD4
      and stat == [3, 1, 2 * 96 + 320, 0, 0], str(stat) + str(CALLS))
check("D audit: the K7 word is left naming the audited layer (the packed run is last)",
      int(FG._S.sfp_word[0]) == mods[1][1].skw["sfp"][1])
twin = FG._S.sfp[1]["dense"][0]
SP.bank_bytes(twin)[3, 1, 77] ^= 1
call(1)
stat = FG._S.sfp_stat.tolist()
check("D audit control that must fail: one flipped code in the dense twin -> mismatch words > 0, control still 0",
      stat[3] > 0 and stat[4] == 0 and stat[0] == 4, str(stat))
SP.bank_bytes(twin)[3, 1, 77] ^= 1
FG.SFP_TICK = 4
rec = []
for i in range(8):
    FG._sfp_tick(types.SimpleNamespace(_emit=rec.append))
check("D tick: one record per SFP_TICK steps with the error word and the audit counters",
      len(rec) == 2 and rec[0]["kernel_err"] == 0 and rec[0]["mismatch_words"] == stat[3]
      and rec[0]["event"] == "sf_pack" and not any("ERROR" in m for m in LOGS))
FG._S.sfp_ctl[1:2].fill_(1)
for i in range(4):
    FG._sfp_tick(None)
check("D tick: a kernel error word is logged as an ERROR, once", sum("K7 packed scales ERROR" in m for m in LOGS) == 1)
for i in range(4):
    FG._sfp_tick(None)
check("D tick: the same error word is not logged again", sum("K7 packed scales ERROR" in m for m in LOGS) == 1)
MM.K7 = MS.config("a2b4")
check("D _sfp_ready: the wrong kernel variant, an unarmed megamoe, or a layer missing are refused",
      raises(lambda: FG._sfp_ready({0, 1, 2}), RuntimeError)
      and (setattr(MM, "K7", MS.config("a2b6")) or setattr(MM, "ACTIVE", False)
           or raises(lambda: FG._sfp_ready({0, 1, 2}), RuntimeError))
      and (setattr(MM, "ACTIVE", True) or raises(lambda: FG._sfp_ready({0, 1, 2, 3}), RuntimeError)))
FG._S.sfp_cfg = ("kernel", "b6", "lo", ())
n0 = len(LOGS)
FG._sfp_ready({0, 1, 2})
check("D kernel-b6: one line, nothing else is built", len(LOGS) == n0 + 1 and "stock path" in LOGS[-1])

# ------------------------------------------------ phase 0 (GRACETIER_SF_PACK=probe) is kept as 0001 had it
FG._S = FG._State()
FG._S.rt = types.SimpleNamespace()
for l in range(2):
    m = Mod()
    m._transformed_l1_weights, m._transformed_l2_weights = (None, mn_bank(3, 128, 4)), (None, mn_bank(3, 128, 2))
    FG._S.k7_modules[l] = m
st = FG._sfpack_probe()
check("D probe mode: 4 banks packed and compared, lossless, one PROBE ONLY line", st["banks"] == 4
      and st["lossless"] is True and st["packed_bytes"] * 2 == st["dense_bytes"] and "PROBE ONLY" in LOGS[-1])

# ================================================================================================ E. residency
def res_world(fmt, order, audit_layer):
    dum, sym = fresh(fmt, order, audit=(audit_layer,))
    lo = SP.FORMATS[fmt].lo
    mods = {l: finalize(l, hot_lo=max(117, lo)) for l in range(3)}
    views, wt = {}, {}
    for l, (m, st, truth, par) in mods.items():
        w = [torch.randint(0, 255, (E_, b), generator=GEN, dtype=torch.uint8) for b in (16384, 8192)]
        gr = [w[k][H_:].clone() for k in (0, 1)]
        wt[l] = [w[k].clone() for k in (0, 1)]
        for k in (0, 1):
            w[k][H_:] = 0                                          # the scratch holds nothing
        row = torch.arange(E_, dtype=torch.int32)
        m._transformed_l1_weights = (w[0], dum[0])
        m._transformed_l2_weights = (w[1], dum[1])
        t = FG._res_tensors(m)
        views[l] = GR.LayerView(l, H_, row.clone(), row.clone(), t[0], t[2], t[1], t[3], gr[0], gr[1])
        if FG._S.sfp[l]["dense"] is not None:
            views[l].also_sf(*FG._S.sfp[l]["dense"])
    cfg = GR.ResConfig(mode="adaptive", freeze=False, dir=None)
    man = GR.GtResManager(views, {l: list(range(H_)) for l in views}, {l: list(range(H_, E_)) for l in views},
                          torch.zeros(3, E_, dtype=torch.int32), cfg, log=lambda s: None)
    return mods, views, man, wt


for fmt, order in (("b6", "lo"), ("b4", "hi")):
    mods, views, man, wt = res_world(fmt, order, 2)
    check(f"E {fmt}/{order}: the manager's SF rows are the packed banks' rows (no dense bank, no dummy)", all(
        v.sf[k].data_ptr() == FG._S.sfp[l]["banks"][k].raw.data_ptr() and v.sf[k].dim() == 2
        for l, v in views.items() for k in (0, 1)) and views[2].sf_also != ((), ()) and views[0].sf_also == ((), ()))
    rng = np.random.default_rng(5)
    swaps = 0
    for _ in range(40):
        man.queue_random(3, rng)
        swaps += man.flush()
    ok, moved = True, 0
    for l, (m, st, truth, par) in mods.items():
        for e in range(E_):
            g = man.row[l][e]
            moved += g != e
            for k, pk in enumerate(FG._S.sfp[l]["banks"]):
                ok &= torch.equal(SP.col_bytes(pk.dense(g, g + 1))[0], truth[k][e])
                if FG._S.sfp[l]["dense"] is not None:
                    ok &= torch.equal(SP.bank_bytes(FG._S.sfp[l]["dense"][k])[g], truth[k][e])
            where, i = man.home(l, e)
            ok &= all(torch.equal(man.weight_bytes(l, e, k), wt[l][k][e]) for k in (0, 1))
    a = man.audit(random_per_layer=E_)
    check(f"E {fmt}/{order}: after {swaps} swaps ({moved} experts displaced) every expert's packed scale row, its "
          f"dense twin and its weights sit in its current group; the manager's own audit finds 0 mismatches",
          ok and swaps > 20 and moved > 6 and a["mismatches"] == 0 and a["checked"] == 3 * E_, str(a["mismatches"]))
    FG._S.sfp[0]["banks"][1].raw[5] ^= 0x10
    a = man.audit(random_per_layer=E_)
    check(f"E {fmt}/{order} control that must fail: one changed packed byte is found by the audit digest",
          a["mismatches"] == 1 and a["first"][0]["parts"] == ["sf2"], str(a.get("first")))
check("E a dense bank still goes through _sf2d as before; a non-int32 or odd packed row matrix is refused",
      GR._sf2d(mn_bank(3, 128, 2), "x").shape == (3, 256)
      and raises(lambda: GR._sf2d(torch.zeros(3, 8, dtype=torch.int16), "x"), ValueError)
      and raises(lambda: GR._sf2d(torch.zeros(3, 7, dtype=torch.int32), "x"), ValueError))

# ================================================================================================ F. a12
sys.modules["native_scales"] = types.SimpleNamespace(NativeScales=object)
A12 = importlib.import_module("fastpath_a12_retire")
GUARD = importlib.import_module("fastpath_a12_memory_guard")
torch.cuda.mem_get_info = lambda d: (1 << 30, 2 << 30)
torch.cuda.memory_reserved = lambda d: 0
torch.cuda.empty_cache = lambda: None
dense_row = dict(layer=0, experts=128, original_logical_bytes=10, candidate_allocated_bytes=12, granted=[1, 1],
                 weights_unchanged=True, original_graphs_captured=0, fallback=False, original_storage_gone=[True, True],
                 status="RETIRED", after=dict(device_free_bytes=1, torch_reserved_bytes=0))


def a12_boot(audit=(), fmt="b6"):
    A12.ROWS.clear()
    del A12.MANAGERS[:]
    dum, sym = fresh(fmt, "lo", audit=audit)
    for l in range(40):
        if l < 3:
            m = finalize(l)[0]
        else:                                  # the other 37 layers: the same record without the bytes
            m = Mod()
            own = (Own(100), Own(50))
            m._k7_sf = {"owners": own, "dense": None, "bits": 6, "order": "lo", "hot_gone": [True, True]}
            m._transformed_l1_weights = (None, dum[0])
        m._mp = types.SimpleNamespace(idx=l, n_hot=384, n_cold=0)
        A12.replace_finalized(m)
        A12.replace_finalized(m)               # a second finalize of the same module is a no-op, as before
    for l in range(40, 43):
        A12.ROWS[l] = dict(dense_row, layer=l)
    d = type("Desc", (), {"cg_mode": types.SimpleNamespace(name="FULL")})()
    mans = []
    for name in ("T", "Q"):
        man = type(name, (), {})()
        man.graphs, man._capture_descs = {d: object()}, {"FULL": [d]}
        mans.append(man)
        A12.MANAGERS.append(dict(manager=name, descriptors=1, graph_keys=[str(d)], original_capture_traversals=1,
                                 runtime_selector=False, original_graphs_captured=0, duplicate_graphs_retained=0))
    sys.modules["vllm.v1.worker.gpu.cudagraph_utils"] = types.SimpleNamespace(
        CUDAGraphMode=types.SimpleNamespace(FULL="FULL"))
    return types.SimpleNamespace(cudagraph_manager=mans[0], speculator=types.SimpleNamespace(
        query_cudagraph_manager=mans[1]))


E_real = FG.E
A12.L1_SHAPE, A12.L2_SHAPE = (N1, Q1), (N2, Q2)
runner = a12_boot()
s = A12.state()
r0 = A12.ROWS[0]
check("F packed layers: a12_retire copies nothing, records the two compressible owners, dense-equivalent logical "
      "bytes and the real allocated bytes", r0["candidate_allocated_bytes"] == sum(o.size for o in FG._S.sfp[0][
          "owners"]) and r0["original_logical_bytes"] == 384 * (N1 * Q1 + N2 * Q2) * 4 and r0["granted"] == [1, 1]
      and r0["status"] == "RETIRED" and r0["original_storage_gone"] == [True, True]
      and r0["k7"]["hot_source_gone"] == [True, True] and r0["k7"]["bits"] == 6)
check("F state(): packed layers listed; no diagnostic banks, no selector", s["k7"]["packed_layers"] == list(range(40))
      and s["diagnostic_banks_bytes"] == 0 and s["runtime_selector"] is False and s["original_banks_retained"] == 0)
check("F memory guard accepts 40 packed + 3 dense layers", GUARD.retirement_facts(runner, A12)["layers"]
      == list(range(43)))
runner = a12_boot(audit=(1,))
s = A12.state()
check("F audit boot: the dense twins are reported as diagnostic banks, one bank retained, a runtime selector",
      s["diagnostic_banks_bytes"] == 384 * (N1 * Q1 + N2 * Q2) * 4 and s["original_banks_retained"] == 1
      and s["runtime_selector"] is True and s["k7"]["audit_layers"] == [1]
      and A12.ROWS[1]["original_storage_gone"] == [False, False])
check("F memory guard refuses dense twins unless the audit is declared", raises(
    lambda: GUARD.retirement_facts(runner, A12), GUARD.Refused))
os.environ["GRACETIER_SF_PACK_AUDIT"] = "1"
check("F memory guard accepts the declared audit", GUARD.retirement_facts(runner, A12)["k7"]["audit_layers"] == [1])
A12.ROWS[5]["original_storage_gone"] = [False, False]
check("F memory guard still refuses an undeclared live dense bank", raises(
    lambda: GUARD.retirement_facts(runner, A12), GUARD.Refused))
del os.environ["GRACETIER_SF_PACK_AUDIT"]
A12.ROWS.clear()
for l in range(43):
    A12.ROWS[l] = dict(dense_row, layer=l)
s = A12.state()
check("F no packed layer: state() has exactly the keys and values it had (no k7 key), and the guard passes",
      "k7" not in s and s["diagnostic_banks_bytes"] == 0 and s["runtime_selector"] is False
      and s["original_banks_retained"] == 0 and GUARD.retirement_facts(runner, A12)["layers"] == list(range(43)))
if BASE:
    sys.modules.pop("fastpath_a12_retire", None)
    BA = load("base_a12", os.path.join(BASE, "tree", "fastpath", "fastpath_a12_retire.py"))
    for l in range(43):
        BA.ROWS[l] = dict(dense_row, layer=l)
    b = BA.state()
    check("F no packed layer: state() equals the unpatched tree's on the same rows", {k: v for k, v in s.items()
          if k != "managers"} == {k: v for k, v in b.items() if k != "managers"})
FG.E = E_real

# ================================================================================================ G. kernel edits
HDR = open(os.path.join(MM.PATCH_DIR, MS.FILE)).read()
counts = [HDR.count(b) for _, _, b, _ in MS.EDITS]
check(f"G each of the {len(MS.EDITS)} edits matches the fork header exactly once", counts == [1] * len(MS.EDITS),
      str(counts))
ok = True
for name, where, b, _ in MS.EDITS:
    first = HDR[:HDR.index(b)].count("\n") + 1
    last = first + b.count("\n") - 1
    ok &= where == (str(first) if first == last else f"{first}-{last}")
check("G every edit's cited line range is where its text is in the unpatched header", ok)
out = {}
for mode, cp in (("a2b6", ""), ("a2b4", ""), ("a2b6", "plain"), ("kernel-b6", "")):
    cfg = MS.config(mode, cp)
    out[(mode, cp)] = MS.render(HDR, cfg)
r6, r4, rp = out[("a2b6", "")], out[("a2b4", "")], out[("a2b6", "plain")]
check("G rendered: braces, parentheses and brackets balance exactly as in the fork header", all(
    r.count(a) - r.count(b) == HDR.count(a) - HDR.count(b) for r in out.values() for a, b in ("{}", "()", "[]")))
check("G rendered: no placeholder left, every K7 name defined before use", all("@" not in r for r in out.values())
      and all(r.index("struct K7 {") < r.index("w6k7") < r.index("k7_on()") for r in out.values()))
check("G b6: bias 64, 384 B a block, the copy carries .b8x16.b6x16_p32", "kK7Bias = 64;" in r6
      and "kK7TxBytes = 384;" in r6 and "tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b6x16_p32 [%0], %1;" in r6)
check("G b4: bias 112, 256 B a block, the copy carries .b8x16.b4x16_p64", "kK7Bias = 112;" in r4
      and "kK7TxBytes = 256;" in r4 and "tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b4x16_p64 [%0], %1;" in r4)
check("G plain: the copy has no decompress qualifier (CUTLASS's own string)",
      'tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;' in rp and ".b8x16" not in rp.split("struct K7Utccp")[1][:300])
check("G kernel-b6 renders the same header as a2b6 (the difference is on the host)", out[("kernel-b6", "")] == r6)
removed = [i + 1 for i, l in enumerate(HDR.split("\n")) if l not in set(r6.split("\n"))]
check("G the variant rewrites 6 lines of the fork header (773, 912, 1021, 1119, 1402, 1403) and adds 90",
      removed == [773, 912, 1021, 1119, 1402, 1403] and r6.count("\n") - HDR.count("\n") == 90, str(removed))
sites = ("k7_bias_word(remote_sf_ptr[j], w6k7)", "allt_head and not k7_a ? &tensor_map_shared_l1_acts_sf",
         "buffer.input_sf_buffer.get_data_buffer(k7_t)", "static_cast<uint8_t>(sf_exp.x + k7_add)",
         "static_cast<uint8_t>(sf_exp.y + k7_add)", "cute::SM100_TMA_2SM_LOAD_3D::copy(", "w6p3::kK7TxBytes * 2",
         "w6p3::K7Utccp::copy(sf_desc, kTmemStartColOfSFB + i * 4)", "ptx::tensor_map_acquire_gpu(k7_tm)")
check("G every activation-scale site, the packed load, its byte count and the decompressing copy are in the variant",
      all(r6.count(s) == 1 for s in sites), str([r6.count(s) for s in sites]))
check("G a changed fork header is refused, not patched blind", raises(
    lambda: MS.render(HDR.replace("kC11Magic = 0x31314336u", "kC11Magic = 0x31314337u"), MS.config("a2b6")),
    RuntimeError) and raises(lambda: MS.render(r6, MS.config("a2b6")), RuntimeError))
check("G config: a bad copy form or mode raises", raises(lambda: MS.config("a2b6", "zip"), RuntimeError)
      and raises(lambda: MS.config("a2"), RuntimeError) and MS.config("a2b4", "plain")["tag"] == "b4p")
# fastpath_megamoe with the variable set: a new include root, the rendered header in it
names = {}
for mode, cp in (("a2b6", ""), ("a2b4", ""), ("a2b6", "plain"), ("probe", ""), ("garbage", "")):
    os.environ["GRACETIER_SF_PACK"], os.environ["GRACETIER_SF_PACK_CP"] = mode, cp
    M2 = load("megamoe_" + mode + cp, os.path.join(FP, "fastpath_megamoe.py"))
    names[(mode, cp)] = M2.ROOT_NAME
    if mode == "a2b6" and cp == "":
        vend = tempfile.mkdtemp()
        os.makedirs(os.path.join(vend, "include", "deep_gemm", "impls"))
        os.makedirs(os.path.join(vend, "include", "deep_gemm", "scheduler"))
        M2.ROOT_BASE = os.path.join(vend, "roots")
        root = M2.build_root(vend)
        got = open(os.path.join(root, MS.FILE)).read()
        sch = open(os.path.join(root, M2.FILES[1]), "rb").read() == open(os.path.join(M2.PATCH_DIR, M2.FILES[1]),
                                                                         "rb").read()
        check("G set: build_root writes the rendered impl header and the untouched scheduler header", got == r6
              and sch and M2.K7["fmt"] == "b6" and os.path.basename(root).startswith("shared-early-k7b6d-"))
        shutil.rmtree(vend)
del os.environ["GRACETIER_SF_PACK"], os.environ["GRACETIER_SF_PACK_CP"]
check("G each variant has its own include root; probe and an unknown value keep the stock root",
      len({names[k] for k in (("a2b6", ""), ("a2b4", ""), ("a2b6", "plain"))} | {MM.ROOT_NAME}) == 4
      and names[("probe", "")] == MM.ROOT_NAME and names[("garbage", "")] == MM.ROOT_NAME, str(names))
if len(sys.argv) > 3:
    with open(sys.argv[3], "w") as f:
        f.write(r6)
    print(f"     wrote the rendered a2b6 header to {sys.argv[3]}")

print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
