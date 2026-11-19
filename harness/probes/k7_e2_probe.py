#!/usr/bin/env python3
"""K7 experiment E2 (BOX ONLY. Written on a Mac with no GPU. Nothing in this file has run on a GPU.)

Question (research/K7-packed-scales.md, option a): once a 4-bit scale column is loaded with a TMA tensor map of
the 4-bit type the expert kernel uses for its weights (CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) and copied to tensor
memory with tcgen05.cp, what bytes does tensor memory hold? Variant a2 needs one zero-extended byte per nibble, in
a known order. This probe measures it, with controls, and prints which kernel variant the result selects.

Stages (each prints the first 64 bytes it saw and ONE verdict line):
  controls  an 8-bit pattern through the same paths (plain stores or a UINT8 TMA load, the plain copy, tcgen05.ld)
            must come back unchanged; every launch also round-trips a pattern through tcgen05.st / tcgen05.ld.
  stage 1   TMA load of a packed 4-bit pattern (every nibble value at known places, and nibbles 5..15) with
            16U4_ALIGN16B and 16U4_ALIGN8B: the shared-memory bytes, the nibble order, the byte count the barrier
            sees.
  stage 2   tcgen05.cp of that shared memory to tensor memory: the plain copy the scale path uses today, and the
            4-bit to 8-bit decompressing copy of the PTX manual (.b8x16.b4x16_p64). Fed by hand-built rows (the
            copy alone) and by the TMA load (the whole route).
  stage 2b  the same with 6-bit codes (16U6_ALIGN16B, .b8x16.b6x16_p32): the fallback if 4 bits cannot be used.
  stage 3   one block-scaled multiply (128 x 16 x 128): weight scales from the packed column, activation scales
            carrying the constant (+112 for 4 bits, +64 for 6 bits), compared bit for bit with dense 8-bit scales.

How it runs: this file compiles k7_e2_kernel.cu (+ k7_e2_ptx.cuh, same directory) with nvcc, once per copy form,
the way Garner's kernels are built in this image (kernels/w8-persist/build.sh:10-12):
    nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC
Each case then runs in its own child process (ctypes, no torch), so a kernel that faults cannot poison the
cases after it. Nothing here reads the checkpoint or needs the server's files.

    python3 k7_e2_probe.py --build-dir /work/scratch/k7-e2-build [--json /work/scratch/k7-e2.json]

Exit code 0 = ran and classified, 2 = could not compile or run, 3 = ran but a control failed (ignore the result).
"""
import argparse
import ctypes
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import time

TAG = "[E2]"
HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_CU = os.path.join(HERE, "k7_e2_kernel.cu")
PTX_CUH = os.path.join(HERE, "k7_e2_ptx.cuh")

# Shared-memory layout of the kernel (k7_e2_kernel.cu; the child checks these against k7e2_abi()).
SF_OFF, SF_BYTES, SFB_OFF, B_OFF, A_OFF, IMAGE_BYTES = 0, 2048, 2048, 4096, 8192, 24576
SENT = 0xEE   # shared memory where a TMA load should land, and tensor memory before the copy
PAD = 0x5A    # the pad bytes of a hand-built padded row (a pad that leaks into the output is visible)
FILL = 0x99   # rows of the global tensor outside the box
NOTW = 0xA5   # output words the kernel never wrote
DONE_MAGIC = 0x4B374532

DT = {"u8": 0, "4a8": 1, "4a16": 2, "6a16": 3}                       # k7e2_enum() index
DT_NAME = {"u8": "CU_TENSOR_MAP_DATA_TYPE_UINT8", "4a8": "CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B",
           "4a16": "CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B", "6a16": "CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B"}
ROW_BYTES = {"u8": 128, "4a8": 64, "4a16": 64, "6a16": 96}          # global bytes per 128 elements
DATA_BYTES = {"4a16": 8, "6a16": 12}                                # packed bytes per 16 elements, rest of 16 is pad
FORM_STR = {0: "tcgen05.cp.cta_group::1.32x128b.warpx4",
            1: "tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b4x16_p64",
            2: "tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b6x16_p32",
            3: "tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64",
            4: "tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32",
            5: "tcgen05.cp.cta_group::1.128x128b"}
FORM_ROWS = {0: 32, 1: 32, 2: 32, 3: 128, 4: 128, 5: 128}
NVCC_FLAGS = ["-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC"]
ORDERS = ("in order (low nibble first)", "NEIGHBOURS SWAPPED (high nibble first)", "REVERSED inside each 16")
_PERM = (lambda j: j, lambda j: j ^ 1, lambda j: 15 - j)


def log(msg=""):
    print(f"{TAG} {msg}" if msg else TAG, flush=True)


# ------------------------------------------------------------------------------------------------ patterns
def lcg(seed):
    x = seed & 0x7FFFFFFF
    while True:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        yield x >> 8


_REAL = (5, 6, 7, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 15)


def nib_all(n):
    """Every nibble value once per 16, at a place that depends on the row (a bijection in each row)."""
    return [(5 * (i % 16) + 7 * (i // 16) + 3) & 15 for i in range(n)]


def nib_real(n, seed=20261119):
    """Nibbles 5..15, bunched like the real codes 117..127 (E0), fixed seed."""
    g = lcg(seed)
    return [_REAL[next(g) % len(_REAL)] for _ in range(n)]


def six_all(n):
    """Every 6-bit value, 16 different ones in each row."""
    return [(37 * i + 11) & 63 for i in range(n)]


def bytes_ctl(n):
    """An 8-bit control pattern with no structure a 4-bit path could reproduce."""
    return [((i * 73 + 41) ^ (i >> 3)) & 0xFF for i in range(n)]


def pack4(q):
    """Two nibbles per byte, even index in the low nibble: gracetier_sfpack.pack_bytes (K7 patch)."""
    return bytes(q[i] | (q[i + 1] << 4) for i in range(0, len(q), 2))


def pack6(v):
    """Sixteen 6-bit values in 12 bytes, value j at bits 6j..6j+5 (little endian)."""
    out = bytearray()
    for i in range(0, len(v), 16):
        x = 0
        for j in range(16):
            x |= v[i + j] << (6 * j)
        out += x.to_bytes(12, "little")
    return bytes(out)


def pack(kind, elems):
    if kind == "u8":
        return bytes(elems)
    return pack6(elems) if kind == "6a16" else pack4(elems)


def padded(packed, db, pad=PAD):
    """Rows of 16 bytes: db packed bytes, then pad (the layout the manual gives for .b4x16_p64 / .b6x16_p32)."""
    return b"".join(packed[i:i + db] + bytes([pad]) * (16 - db) for i in range(0, len(packed), db))


def rows16(b, rows):
    return [list(b[16 * r:16 * r + 16]) for r in range(rows)]


def hexrows(b, n=64):
    b = bytes(b[:n])
    return [f"{o:04x}: " + " ".join(f"{x:02x}" for x in b[o:o + 16]) for o in range(0, len(b), 16)]


def lanes(hexstr, cols):
    """Tensor-memory readback -> 128 byte strings (lane 0..127), cols * 4 bytes each."""
    raw = bytes.fromhex(hexstr)
    n = cols * 4
    return [raw[i * n:(i + 1) * n] for i in range(128)]


def sw128_offset(row, kbyte):
    """Byte offset of (row, k) in a K-major SWIZZLE_128B tile with 128-byte rows (W3 ptx.cuh:207-209)."""
    return (row >> 3) * 1024 + (row & 7) * 128 + ((((kbyte >> 4) ^ (row & 7)) & 7) << 4) + (kbyte & 15)


def unswizzle128(smem, box_rows):
    """Undo SWIZZLE_128B on a box of box_rows rows of 128 bytes (only used if the plain map is refused)."""
    out = bytearray(smem)
    for r in range(box_rows):
        for j in range(8):
            src = r * 128 + ((j ^ (r & 7)) << 4)
            out[r * 128 + 16 * j:r * 128 + 16 * j + 16] = smem[src:src + 16]
    return bytes(out)


# ------------------------------------------------------------------------------------------------ classification
def find_map(out_rows, elem_rows):
    """Is every output byte ONE function of the element at the same place (or at a simple reordering)?
    Returns (order index, {element value: byte}) or None. The function must tell values apart (injective)."""
    for k, perm in enumerate(_PERM):
        m, ok = {}, True
        for o, e in zip(out_rows, elem_rows):
            for j in range(16):
                if m.setdefault(e[perm(j)], o[j]) != o[j]:
                    ok = False
                    break
            if not ok:
                break
        if ok and len(m) > 1 and len(set(m.values())) == len(m):
            return k, m
    return None


def map_name(m):
    ks = {b - v for v, b in m.items()}
    if ks == {0}:
        return "ZERO-EXTENDED", "byte = value"
    if len(ks) == 1:
        k = ks.pop()
        return f"OFFSET{k:+d}", f"byte = value {k:+d}" + (" (= 0x70 | value)" if k == 112 else "")
    for s in (1, 2, 3, 4):
        if all(b == ((v << s) & 0xFF) for v, b in m.items()):
            return f"SHIFTED<<{s}", f"byte = value << {s} (value * {1 << s})"
    return "OTHER", "byte = " + " ".join(f"{v}:{b:02x}" for v, b in sorted(m.items())[:16])


def offset_of(c):
    """The constant k if tensor / shared memory holds value + k for every element (0 = zero-extended), else None."""
    if not c or c.get("cls") != "MAP":
        return None
    if c.get("map") == "ZERO-EXTENDED":
        return 0
    return int(c["map"][6:]) if str(c.get("map", "")).startswith("OFFSET") else None


def classify_smem(kind, smem, packed, elems, rows):
    """What a TMA load left in shared memory. Returns {'cls', 'text', ...}."""
    n = rows * 16
    if all(b == NOTW for b in smem[:n]):
        return {"cls": "NODUMP", "text": "the kernel never wrote the dump"}
    if all(b == SENT for b in smem[:n]):
        return {"cls": "UNTOUCHED", "text": "the load wrote nothing (the sentinel is intact)"}
    if kind == "u8":
        if bytes(smem[:n]) == bytes(packed[:n]):
            return {"cls": "RAW", "text": "byte for byte the box in global memory"}
        return {"cls": "UNCLASSIFIED", "text": "NOT the box in global memory"}
    if kind in DATA_BYTES:
        db = DATA_BYTES[kind]
        if all(bytes(smem[16 * r:16 * r + db]) == bytes(packed[db * r:db * (r + 1)]) for r in range(rows)):
            pads = b"".join(bytes(smem[16 * r + db:16 * r + 16]) for r in range(rows))
            pad = ("untouched" if all(b == SENT for b in pads) else
                   "zero" if not any(pads) else f"other bytes (first {pads[:8].hex()})")
            return {"cls": "PACKED+PAD", "text": f"STILL PACKED, {db} data + {16 - db} pad bytes per 16 elements, "
                                                 f"low element first; pad {pad}"}
    fm = find_map(rows16(smem, rows), rows16(elems, rows))
    if fm is not None:
        name, how = map_name(fm[1])
        return {"cls": "MAP", "map": name, "order": fm[0],
                "text": f"ONE BYTE PER ELEMENT, {ORDERS[fm[0]]}, {how}"}
    m = len(packed)
    if bytes(smem[:m]) == bytes(packed):
        rest = "untouched" if all(b == SENT for b in smem[m:n]) else "changed"
        return {"cls": "PACKED", "text": f"STILL PACKED, no gaps ({m} bytes, low element first); the rest {rest}"}
    return {"cls": "UNCLASSIFIED", "text": "matches no layout this probe knows (read the dump)"}


def classify_tmem(res, smem, elems, form_rows):
    """What tcgen05.cp left in tensor-memory columns 0..3. Returns {'cls', 'text', 'quad', 'hi'}."""
    ln = lanes(res["tmem"], 8)
    sf, hi = [x[:16] for x in ln], [x[16:] for x in ln]
    out = {"quad": all(sf[i] == sf[i % 32] for i in range(128)), "hi": all(h == bytes([SENT]) * 16 for h in hi)}
    base = sf[:form_rows]
    if all(b == NOTW for x in base for b in x):
        out.update(cls="NODUMP", text="the kernel never wrote the readback")
    elif all(x == bytes([SENT]) * 16 for x in base):
        out.update(cls="UNTOUCHED", text="the copy wrote nothing (the sentinel is intact)")
    elif all(base[r] == bytes(smem[16 * r:16 * r + 16]) for r in range(form_rows)):
        out.update(cls="RAW", text="RAW COPY of shared memory (16 bytes per lane, nothing converted)")
    else:
        fm = find_map([list(x) for x in base], rows16(elems, form_rows))
        if fm is not None:
            name, how = map_name(fm[1])
            out.update(cls="MAP", map=name, order=fm[0], table={str(v): b for v, b in fm[1].items()},
                       text=f"ONE BYTE PER ELEMENT, {ORDERS[fm[0]]}, {how}")
        else:
            out.update(cls="UNCLASSIFIED", text="matches no layout this probe knows (read the dump)")
    return out


def ctl_ok(res):
    """The tcgen05.st -> tcgen05.ld round trip of step 1."""
    try:
        ln = lanes(res["ctl"], 8)
        return all(struct.unpack("<8I", ln[r]) == tuple(0xC0DE0000 | (r << 8) | c for c in range(8))
                   for r in range(128))
    except Exception:  # noqa: BLE001
        return False


def res_ok(res):
    return res.get("rc") == 0 and not res.get("exception")


def res_error(res):
    if res.get("exception"):
        return str(res["exception"])
    rc = res.get("rc")
    s = {1: "bad argument", 2: "cuTensorMapEncodeTiled refused the map", 3: "device allocation failed",
         4: "the launch failed", 5: "the kernel failed", 6: "a device copy failed"}.get(rc, f"rc {rc}")
    if rc == 2:
        ee = res.get("encode_err")
        s += {-1: " (no cuTensorMapEncodeTiled entry point)",
              -2: " (this data type name is not in the image's cuda.h)"}.get(ee, f" (CUresult {ee})")
    if res.get("cuda_err"):
        s += f" (cudaError {res.get('cuda_err')}: {res.get('err')})"
    return s


# ------------------------------------------------------------------------------------------------ one case
class Ctx:
    def __init__(self, execf, libs, skip_mma=False):
        self.execf, self.libs, self.skip_mma = execf, libs, skip_mma
        self.cases, self.verdicts = [], []

    def verdict(self, text):
        self.verdicts.append(text)
        log("VERDICT " + text)


def make_req(lib, form, image, src=None, kind="u8", swizzle=0, box_rows=4, c1=4, expect_tx=0, do_mma=0):
    return {"lib": lib, "form": form, "image": bytes(image).hex(), "src": None if src is None else bytes(src).hex(),
            "src_mode": 0 if src is None else 1, "dtype": DT[kind], "kind": kind, "swizzle": swizzle, "dim0": 128,
            "dim1": 32, "stride1": ROW_BYTES[kind], "box0": 128, "box1": box_rows, "c0": 0, "c1": c1,
            "expect_tx": expect_tx, "do_mma": do_mma}


def sf_image(content, fill=SENT):
    image = bytearray(IMAGE_BYTES)
    image[SF_OFF:SF_OFF + SF_BYTES] = bytes([fill]) * SF_BYTES
    image[SF_OFF:SF_OFF + len(content)] = content
    return image


def global_tensor(kind, packed, c1):
    g = bytearray([FILL]) * (ROW_BYTES[kind] * 32)
    g[c1 * ROW_BYTES[kind]:c1 * ROW_BYTES[kind] + len(packed)] = packed
    return bytes(g)


def launch(ctx, form, kind, elems, mode, image=None, do_mma=0):
    """Run one launch. mode: 'raw' (plain stores of the element bytes), 'pad' (plain stores of hand-built padded
    rows), 'tma' (a TMA load of the packed elements). Returns (res, meta); for 'tma' it finds the byte count the
    barrier accepts and falls back to SWIZZLE_128B if the plain map is refused."""
    rows = FORM_ROWS[form]
    lib = ctx.libs[form]
    packed = pack(kind, elems)
    meta = {"form": form, "kind": kind, "mode": mode, "rows": rows, "packed": packed, "elems": elems, "tx": None,
            "swizzle": 0, "tx_tried": []}
    if mode != "tma":
        content = bytes(elems) if mode == "raw" else padded(packed, DATA_BYTES[kind])
        img = image if image is not None else sf_image(content)
        if image is not None:
            img[SF_OFF:SF_OFF + SF_BYTES] = sf_image(content)[SF_OFF:SF_OFF + SF_BYTES]
        res = ctx.execf(make_req(lib, form, img, do_mma=do_mma))
        return res, meta
    box_rows = rows // 8
    src = global_tensor(kind, packed, box_rows)
    img = image if image is not None else sf_image(b"")
    if image is not None:
        img[SF_OFF:SF_OFF + SF_BYTES] = bytes([SENT]) * SF_BYTES
    cands = [box_rows * ROW_BYTES[kind]] + ([box_rows * 128] if ROW_BYTES[kind] != 128 else [])
    first = None
    for swz in (0, 1):
        refused = False
        for tx in cands:
            res = ctx.execf(make_req(lib, form, img, src=src, kind=kind, swizzle=swz, box_rows=box_rows, c1=box_rows,
                                     expect_tx=tx, do_mma=do_mma))
            meta["tx_tried"].append(tx)
            if res.get("rc") == 2:
                refused = True
                first = first or res
                break
            if not res_ok(res):
                return res, meta
            meta["swizzle"] = swz
            if res["status"][0] == 1:
                meta["tx"] = tx
                return res, meta
            first = first if (first is not None and res_ok(first)) else res
        if not refused:
            break
    return first, meta


def sf_case(ctx, name, form, kind, elems, mode, quiet_tmem=False):
    """One scale-path case: launch, print the dumps, classify shared memory (TMA only) and tensor memory."""
    log(f"---- {name}")
    log(f"     copy: {FORM_STR[form]}; source: " +
        ("plain stores of the bytes" if mode == "raw" else
         f"plain stores of hand-built padded rows ({DATA_BYTES[kind]} packed + {16 - DATA_BYTES[kind]} pad)"
         if mode == "pad" else f"TMA load, {DT_NAME[kind]}, box 128 elements x {FORM_ROWS[form] // 8} rows"))
    out = {"name": name, "form": form, "kind": kind, "mode": mode, "ok": False, "smem_cls": None, "tmem": None}
    try:
        res, meta = launch(ctx, form, kind, elems, mode)
    except Exception as e:  # noqa: BLE001
        res, meta = {"rc": -7, "exception": f"{type(e).__name__}: {e}"}, {"rows": FORM_ROWS[form], "tx_tried": []}
    out["res"] = {k: v for k, v in res.items() if k not in ("ctl", "acc")}
    ctx.cases.append(out)
    if not res_ok(res):
        ctx.verdict(f"[{name}]: COULD NOT RUN: {res_error(res)}")
        return out
    rows, st = meta["rows"], res["status"]
    smem = bytes.fromhex(res["smem"])
    tm = lanes(res["tmem"], 8)
    tm_first = b"".join(x[:16] for x in tm[:4])
    if mode == "tma":
        log("     global (the box, packed):")
        for line in hexrows(meta["packed"]):
            log("       " + line)
    log("     shared memory after the source step:")
    for line in hexrows(smem):
        log("       " + line)
    log("     tensor memory after the copy (lanes 0..3, columns 0..3):")
    for line in hexrows(tm_first):
        log("       " + line)
    parts = ["st/ld round trip " + ("ok" if ctl_ok(res) else "FAILED")]
    if mode == "tma":
        parts.append((f"TMA barrier completed with {meta['tx']} bytes expected" if meta["tx"] is not None else
                      f"TMA barrier DID NOT COMPLETE for {meta['tx_tried']} bytes") +
                     (" (SWIZZLE_128B map: the plain map was refused)" if meta["swizzle"] else ""))
    parts.append("copy barrier " + ("completed" if st[2] == 1 else "DID NOT COMPLETE"))
    parts.append("kernel reached its end" if st[7] == DONE_MAGIC else "KERNEL DID NOT REACH ITS END")
    log("     " + "; ".join(parts))
    out.update(ok=True, ctl=ctl_ok(res), tx=meta["tx"], swizzle=meta["swizzle"], copy_done=st[2] == 1,
               ended=st[7] == DONE_MAGIC)
    if mode == "tma":
        view = unswizzle128(smem, rows // 8) if meta["swizzle"] else smem
        sc = classify_smem(kind, view, meta["packed"], meta["elems"], rows)
        out["smem_cls"] = sc
        tx = f"; barrier counts {meta['tx']} bytes" if meta["tx"] is not None else \
            "; BYTE COUNT UNKNOWN (no candidate completed the barrier)"
        ctx.verdict(f"smem [{name}]: {sc['text']}{tx}")
    tc = classify_tmem(res, smem, meta["elems"], rows)
    out["tmem"] = tc
    odd = ([] if tc["quad"] or rows != 32 else ["the 4 lane quadrants DIFFER"]) + \
          ([] if tc["hi"] else ["columns 4..7 CHANGED"])
    if not quiet_tmem:
        ctx.verdict(f"tmem [{name}]: {tc['text']}" + ("; BUT " + ", ".join(odd) if odd else ""))
    return out


def same_map(a, b):
    return bool(a) and bool(b) and a.get("cls") == "MAP" and b.get("cls") == "MAP" and \
        a.get("map") == b.get("map") and a.get("order") == b.get("order")


# ------------------------------------------------------------------------------------------------ stage 3 helpers
def act_code(n, b):
    return 118 + ((3 * n + 5 * b) % 9)          # 118..126: +112 stays below 255


def sfb_atom(shift):
    """Activation scale atom (512 B): token n, K block b at byte n * 16 + b (N = 16 tokens)."""
    a = bytearray([0x7F]) * 512
    for n in range(16):
        for b in range(4):
            a[n * 16 + b] = act_code(n, b) + shift
    return bytes(a)


def mma_image(data, shift, unit=False):
    """Operands of the multiply: weights 128 x 128 and activations 16 x 128, e4m3, K-major SWIZZLE_128B."""
    img = bytearray(IMAGE_BYTES)
    img[SFB_OFF:SFB_OFF + 512] = bytes([0x7F]) * 512 if unit else sfb_atom(shift)
    g = lcg(777)

    def val():
        if data == "ones":
            return 0x38                           # e4m3 1.0
        x = next(g)
        return ((x & 1) << 7) | ((3 + (x >> 1) % 8) << 3) | ((x >> 5) & 7)   # exponent field 3..10: no NaN

    for r in range(128):
        for k in range(128):
            img[A_OFF + sw128_offset(r, k)] = val()
    for r in range(16):
        for k in range(128):
            img[B_OFF + sw128_offset(r, k)] = val()
    return img


def oracle_ones(codes):
    """128 x 16 fp32 words the multiply must give for all-ones data: 32 * sum_b 2^(w(m,b) + a(n,b) - 254).
    codes: the dense weight scale atom (512 B), row m, block b at (m % 32) * 16 + (m // 32) * 4 + b."""
    out = []
    for m in range(128):
        for n in range(16):
            x = 0.0
            for b in range(4):
                w = codes[(m % 32) * 16 + (m // 32) * 4 + b]
                x += 32.0 * 2.0 ** (w + act_code(n, b) - 254)
            bits = struct.pack("<f", x)
            if struct.unpack("<f", bits)[0] != x:
                raise RuntimeError("oracle value is not exact in fp32")
            out.append(bits)
    return b"".join(out)


def acc_bytes(res):
    return bytes.fromhex(res["acc"])


def acc_cmp(a, b):
    n = sum(1 for i in range(0, len(a), 4) if a[i:i + 4] != b[i:i + 4])
    return {"equal": a == b, "words_differ": n, "words": len(a) // 4}


# ------------------------------------------------------------------------------------------------ the experiment
def experiment(ctx):
    S = {"exit_code": 0, "decision": None}
    L = ctx.libs

    log("==== controls: an 8-bit pattern through the same paths must come back unchanged")
    c1 = sf_case(ctx, "control A: stores > plain cp", 0, "u8", bytes_ctl(512), "raw")
    if not c1["ok"]:
        log("could not run: the first control case did not run (see the line above)")
        S.update(exit_code=2, decision="NOT RUN")
        return S
    dev = c1["res"].get("device")
    if dev:
        log(f"     device: {dev}")
    c2 = sf_case(ctx, "control B: TMA u8 > plain cp", 0, "u8", bytes_ctl(512), "tma")

    def good(c, need_tma):
        t = c.get("tmem") or {}
        ok = c["ok"] and c["ctl"] and c["copy_done"] and c["ended"] and t.get("cls") == "RAW" and t.get("quad") \
            and t.get("hi")
        if need_tma:
            ok = ok and (c.get("smem_cls") or {}).get("cls") == "RAW" and c.get("tx") is not None
        return bool(ok)

    S["ctl_manual"], S["ctl_tma"] = good(c1, False), good(c2, True)
    ctx.verdict("controls: readback path (plain stores, plain copy, tcgen05.ld) " +
                ("PASS" if S["ctl_manual"] else "FAIL") + "; TMA path (UINT8 load) " +
                ("PASS" if S["ctl_tma"] else "FAIL"))

    n32 = 512
    log("==== stage 1: TMA load of a packed 4-bit column into shared memory")
    t4 = sf_case(ctx, "TMA 4-bit ALIGN16B > plain cp, all values", 0, "4a16", nib_all(n32), "tma")
    t4r = sf_case(ctx, "TMA 4-bit ALIGN16B > plain cp, nibbles 5..15", 0, "4a16", nib_real(n32), "tma", quiet_tmem=True)
    t8 = sf_case(ctx, "TMA 4-bit ALIGN8B > plain cp, all values", 0, "4a8", nib_all(n32), "tma", quiet_tmem=True)
    S["t4"], S["t4_real"], S["t4a8"] = t4.get("smem_cls"), t4r.get("smem_cls"), t8.get("smem_cls")
    S["t4_plain_tmem"] = t4.get("tmem")

    log("==== stage 2: tcgen05.cp with the 4-bit to 8-bit decompress qualifier")
    f4 = 1 if 1 in L else 3 if 3 in L else None
    S["form4"] = f4
    if f4 is None:
        ctx.verdict("decompressing copy (4-bit): NOT AVAILABLE, the assembler refused every form (see the compile "
                    "lines)")
    else:
        if f4 == 3:
            log("     form 1 (32x128b.warpx4) was refused; using the 128x128b shape, with its own plain control")
            if 5 in L:
                c5 = sf_case(ctx, "control C: stores > plain 128x128b cp", 5, "u8", bytes_ctl(2048), "raw")
                S["ctl_128"] = bool(c5["ok"] and (c5.get("tmem") or {}).get("cls") == "RAW")
        n = FORM_ROWS[f4] * 16
        d4 = sf_case(ctx, "padded rows > 4-bit cp, all values", f4, "4a16",
                     nib_all(n), "pad")
        p4 = sf_case(ctx, "TMA 4-bit ALIGN16B > 4-bit cp, all values", f4, "4a16", nib_all(n),
                     "tma")
        p4r = sf_case(ctx, "TMA 4-bit ALIGN16B > 4-bit cp, nibbles 5..15", f4, "4a16", nib_real(n), "tma")
        S["cp4"], S["pipe4"], S["pipe4_real"] = d4.get("tmem"), p4.get("tmem"), p4r.get("tmem")

    log("==== stage 2b: the same with 6-bit codes (the fallback if 4 bits cannot be used)")
    f6 = 2 if 2 in L else 4 if 4 in L else None
    S["form6"] = f6
    if f6 is None:
        ctx.verdict("decompressing copy (6-bit): NOT AVAILABLE, the assembler refused every form")
    else:
        n = FORM_ROWS[f6] * 16
        d6 = sf_case(ctx, "padded rows > 6-bit cp, all values", f6, "6a16",
                     six_all(n), "pad")
        p6 = sf_case(ctx, "TMA 6-bit ALIGN16B > 6-bit cp, all values", f6, "6a16", six_all(n),
                     "tma")
        p6r = sf_case(ctx, "TMA 6-bit ALIGN16B > 6-bit cp, codes-64", f6, "6a16",
                      [48 + v for v in nib_real(n)], "tma")
        S["cp6"], S["pipe6"], S["pipe6_real"] = d6.get("tmem"), p6.get("tmem"), p6r.get("tmem")
        S["t6"] = p6.get("smem_cls")

    if not ctx.skip_mma:
        stage3(ctx, S)
    decide(ctx, S)
    return S


def stage3(ctx, S):
    """One block-scaled multiply per route, against dense 8-bit scales."""
    log("==== stage 3: one block-scaled multiply, weight scales from the packed column")
    L = ctx.libs
    dense = [0x70 | v for v in nib_real(512, seed=424242)]     # the true codes 0x75..0x7F, in storage order
    nibs, sixes = [c & 15 for c in dense], [c - 64 for c in dense]
    S["mma"] = {}

    def run(name, form, kind, elems, mode, data, shift, unit=False):
        try:
            res, meta = launch(ctx, form, kind, elems, mode, image=mma_image(data, shift, unit), do_mma=1)
        except Exception as e:  # noqa: BLE001
            res = {"rc": -7, "exception": f"{type(e).__name__}: {e}"}
        if not res_ok(res):
            log(f"     {name} ({data}): could not run: {res_error(res)}")
            return None
        if res["status"][4] != 1:
            log(f"     {name} ({data}): the multiply barrier did not complete")
        return acc_bytes(res)

    unit = run("unit", 0, "u8", [0x7F] * 512, "raw", "ones", 0, unit=True)
    unit_ok = unit is not None and unit == struct.pack("<f", 128.0) * (128 * 16)
    ref1 = run("dense", 0, "u8", dense, "raw", "ones", 0)
    refr = run("dense", 0, "u8", dense, "raw", "rand", 0)
    want = oracle_ones(dense)
    oracle_ok = ref1 is not None and ref1 == want
    neg = run("dense, activation scales + 1", 0, "u8", dense, "raw", "rand", 1)
    neg_ok = refr is not None and neg is not None and neg != refr
    nz = 0.0 if refr is None else sum(1 for i in range(0, len(refr), 4) if refr[i:i + 4] != b"\0\0\0\0") / 2048.0
    S["mma_harness"] = bool(unit_ok and oracle_ok and neg_ok and nz > 0.5)
    ctx.verdict("multiply harness: " + ("PASS" if S["mma_harness"] else "FAIL (stage 3 is not trusted)") +
                ": unit scales give 128.0 " + ("yes" if unit_ok else "NO") +
                ", dense scales give the exact value " + ("yes" if oracle_ok else "NO") +
                ", a changed scale changes the result " + ("yes" if neg_ok else "NO") + f", non-zero words {nz:.2f}")

    routes = []

    def route(label, form, kind, elems, cls, base, key, mode):
        """cls is what stage 2 found for this route. Follow it: neighbours found swapped are packed swapped, and a
        constant that tensor memory already carries comes off the activation shift. The multiply tests that reading."""
        how, k = "", offset_of(cls) or 0
        if cls and cls.get("cls") == "MAP" and cls.get("order") == 1:
            elems, how = [elems[i ^ 1] for i in range(len(elems))], ", neighbours swapped at pack"
        shift = base - k
        if not (0 <= 118 + shift and 126 + shift <= 254):
            log(f"     {label}: skipped, an activation shift of {shift:+d} does not fit a scale code")
            return
        routes.append((f"{label}, act {shift:+d}{how}", form, kind, elems, mode, shift, key))

    if S.get("form4") == 1:
        tma4 = (S.get("t4") or {}).get("cls") == "PACKED+PAD" and S.get("ctl_tma")
        route("4-bit decompress", 1, "4a16", nibs, S.get("pipe4") if tma4 else S.get("cp4"), 112,
              "4bit-decomp", "tma" if tma4 else "pad")
    if (S.get("t4") or {}).get("cls") == "MAP":
        route("4-bit plain", 0, "4a16", nibs, S.get("t4"), 112, "4bit-plain", "tma")
    if S.get("form6") == 2:
        tma6 = (S.get("t6") or {}).get("cls") == "PACKED+PAD" and S.get("ctl_tma")
        route("6-bit decompress", 2, "6a16", sixes, S.get("pipe6") if tma6 else S.get("cp6"), 64,
              "6bit-decomp", "tma" if tma6 else "pad")
    for name, form, kind, elems, mode, shift, key in routes:
        if form not in L:
            continue
        a1 = run(name, form, kind, elems, mode, "ones", shift)
        ar = run(name, form, kind, elems, mode, "rand", shift)
        r = {"ones_exact": a1 is not None and a1 == want, "rand_equal": ar is not None and refr is not None and
             ar == refr, "source": "TMA" if mode == "tma" else "hand-built rows",
             "rand": None if ar is None or refr is None else acc_cmp(ar, refr)}
        S["mma"][key] = r
        d = [] if r["rand"] is None or r["rand"]["equal"] else \
            [f"{r['rand']['words_differ']} of {r['rand']['words']} words"]
        if a1 is not None and not r["ones_exact"]:
            d.append("first all-ones word wanted " + want[:4].hex() + " got " + a1[:4].hex())
        ctx.verdict(f"multiply [{name}, via {r['source']}]: all-ones exact " + ("yes" if r["ones_exact"] else "NO") +
                    ", random data vs dense 8-bit scales " + ("IDENTICAL" if r["rand_equal"] else "DIFFERENT") +
                    (" (" + "; ".join(d) + ")" if d else ""))


def decide(ctx, S):
    """The decision table of research/K7-E2-design.md."""
    t4, cp4, pipe4 = S.get("t4"), S.get("cp4"), S.get("pipe4")
    cp6, pipe6 = S.get("cp6"), S.get("pipe6")
    mma = S.get("mma") or {}
    why, dec, key = [], None, None

    def confirmed(k):
        """None = not measured, True / False = what the multiply said."""
        if not S.get("mma_harness") or k not in mma:
            return None
        return bool(mma[k]["rand_equal"] and mma[k]["ones_exact"])

    def usable(c):
        """Tensor memory holds value + constant for every element, in order or with neighbours swapped."""
        return offset_of(c) is not None and c.get("order") in (0, 1)

    def a2(c, copy, base):
        k = offset_of(c)
        if k:
            why.append(f"tensor memory already holds value {k:+d}: the activations need {base - k:+d}, not +{base}")
        return ("A2 AS DESIGNED" if c["order"] == 0 else "A2 WITH THE NIBBLES SWAPPED AT PACK TIME") + f" ({copy})"

    if not S.get("ctl_manual"):
        dec = "NO DECISION"
        why.append("the readback control failed, so no tensor-memory dump of this run can be trusted")
    else:
        if not S.get("ctl_tma"):
            why.append("the TMA control failed: lines that depend on a TMA load are not trusted")
        if S.get("ctl_tma") and usable(t4) and (S.get("t4_plain_tmem") or {}).get("cls") == "RAW":
            dec, key = a2(t4, "plain copy", 112), "4bit-plain"     # the load itself gives one byte per nibble
        elif S.get("ctl_tma") and usable(pipe4) and S.get("form4") == 1:
            dec, key = a2(pipe4, "decompressing copy", 112), "4bit-decomp"
            if not same_map(S.get("pipe4_real"), pipe4):
                why.append("WARNING: the nibbles 5..15 pattern did not classify the same way as the all-values "
                           "pattern")
        else:
            if pipe4 and pipe4.get("cls") == "MAP":
                why.append(f"4 bits: tensor memory holds {pipe4['text']}; that is not the code plus a constant, so "
                           "the multiply cannot use it")
            elif cp4 and cp4.get("cls") == "MAP":
                why.append(f"4 bits: the decompressing copy alone gives {cp4['text']}; the TMA route did not classify")
            elif S.get("form4") is None:
                why.append("4 bits: no decompressing copy assembled")
            else:
                why.append("4 bits: no route gave one byte per nibble holding the value plus a constant")
            if S.get("form4") == 3:
                why.append("4 bits: the decompressing copy exists only in the 128x128b shape, which the scale layout "
                           "of the kernel (32x128b.warpx4) cannot use as is")
            if S.get("ctl_tma") and usable(pipe6) and S.get("form6") == 2:
                dec = "4-BIT A2 IS DEAD; A2 WITH 6-BIT CODES IS ALIVE (frees 3.95 GiB, not 7.91); a1 or a0 for the rest"
                key = "6bit-decomp"
                k = offset_of(pipe6)
                if k:
                    why.append(f"6 bits: tensor memory already holds value {k:+d}: the activations need {64 - k:+d}")
                if pipe6.get("order") == 1:
                    why.append("6 bits: neighbours come out swapped, so pack them swapped")
                if not same_map(S.get("pipe6_real"), pipe6):
                    why.append("WARNING: the 6-bit codes pattern did not classify the same way as the all-values "
                               "pattern")
            else:
                dec = "NEEDS A1 OR A0 (no packed column reaches tensor memory as usable codes)"
                if cp6 and cp6.get("cls") == "MAP":
                    why.append(f"6 bits: the decompressing copy alone gives {cp6['text']}")
        if key is not None:
            c = confirmed(key)
            if c is True:
                why.append("the multiply confirms it: bit-identical to dense 8-bit scales")
            elif c is False:
                why.append("BUT the multiply does NOT confirm it: stage 2 and stage 3 disagree, do not build on this")
                dec = "NO DECISION (the dumps and the multiply disagree)"
            else:
                why.append("the multiply did not run or its harness failed: the decision rests on the dumps alone")
    S["decision"], S["why"] = dec, why
    if S["exit_code"] == 0 and (dec.startswith("NO DECISION") or not S.get("ctl_tma")):
        S["exit_code"] = 3
    log("==== summary")
    for v in ctx.verdicts:
        log("  " + v)
    for w in why:
        log("  note: " + w)
    log("DECISION: " + dec)


# ------------------------------------------------------------------------------------------------ child process
class K7Req(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("image", "src", "out_ctl", "out_smem", "out_tmem", "out_acc",
                                               "out_status")] + \
               [(n, ctypes.c_longlong) for n in ("src_bytes", "src_mode", "tma_dtype", "tma_swizzle", "dim0", "dim1",
                                                 "stride1", "box0", "box1", "c0", "c1", "expect_tx", "do_mma",
                                                 "cuda_err", "encode_err")]


def child_main(req_path, res_path):
    """Run ONE launch through the compiled library and write the raw outputs as JSON."""
    res = {"rc": None}
    try:
        with open(req_path) as f:
            req = json.load(f)
        lib = ctypes.CDLL(req["lib"])
        lib.k7e2_abi.restype, lib.k7e2_abi.argtypes = ctypes.c_longlong, [ctypes.c_int]
        lib.k7e2_run.restype, lib.k7e2_run.argtypes = ctypes.c_longlong, [ctypes.POINTER(K7Req)]
        lib.k7e2_err_string.restype, lib.k7e2_err_string.argtypes = ctypes.c_char_p, [ctypes.c_longlong]
        lib.k7e2_cp_string.restype = ctypes.c_char_p
        lib.k7e2_device.restype = ctypes.c_longlong
        lib.k7e2_device.argtypes = [ctypes.c_char_p, ctypes.c_longlong] + [ctypes.POINTER(ctypes.c_longlong)] * 3
        abi = [int(lib.k7e2_abi(i)) for i in range(10)]
        want = [ctypes.sizeof(K7Req), IMAGE_BYTES, SF_BYTES, req["form"], FORM_ROWS[req["form"]], abi[5], SF_OFF,
                SFB_OFF, B_OFF, A_OFF]
        if abi != want:
            raise RuntimeError(f"library layout {abi} is not what this script expects {want}")
        res["cp_string"] = lib.k7e2_cp_string().decode()
        image = (ctypes.c_ubyte * IMAGE_BYTES).from_buffer_copy(bytes.fromhex(req["image"]))
        src_b = bytes.fromhex(req["src"]) if req.get("src") else b""
        src = (ctypes.c_ubyte * max(1, len(src_b))).from_buffer_copy(src_b or b"\0")
        o_ctl, o_smem = (ctypes.c_uint32 * (128 * 8))(), (ctypes.c_uint32 * (SF_BYTES // 4))()
        o_tmem, o_acc, o_st = (ctypes.c_uint32 * (128 * 8))(), (ctypes.c_uint32 * (128 * 16))(), (ctypes.c_uint32 * 16)()
        q = K7Req(image=ctypes.addressof(image), src=ctypes.addressof(src) if src_b else None,
                  out_ctl=ctypes.addressof(o_ctl), out_smem=ctypes.addressof(o_smem),
                  out_tmem=ctypes.addressof(o_tmem), out_acc=ctypes.addressof(o_acc),
                  out_status=ctypes.addressof(o_st), src_bytes=len(src_b), src_mode=req["src_mode"],
                  tma_dtype=req["dtype"], tma_swizzle=req["swizzle"], dim0=req["dim0"], dim1=req["dim1"],
                  stride1=req["stride1"], box0=req["box0"], box1=req["box1"], c0=req["c0"], c1=req["c1"],
                  expect_tx=req["expect_tx"], do_mma=req["do_mma"], cuda_err=0, encode_err=0)
        rc = int(lib.k7e2_run(ctypes.byref(q)))
        res.update(rc=rc, cuda_err=int(q.cuda_err), encode_err=int(q.encode_err),
                   err=lib.k7e2_err_string(q.cuda_err).decode(errors="replace") if q.cuda_err else "",
                   status=list(o_st), ctl=bytes(o_ctl).hex(), smem=bytes(o_smem).hex(), tmem=bytes(o_tmem).hex(),
                   acc=bytes(o_acc).hex())
        name = ctypes.create_string_buffer(256)
        mj, mn, tot = ctypes.c_longlong(0), ctypes.c_longlong(0), ctypes.c_longlong(0)
        if lib.k7e2_device(name, 256, ctypes.byref(mj), ctypes.byref(mn), ctypes.byref(tot)) == 0:
            res["device"] = f"{name.value.decode(errors='replace')}, compute capability {mj.value}.{mn.value}, " \
                            f"{tot.value / 2 ** 30:.1f} GiB"
    except BaseException as e:  # noqa: BLE001 - the parent prints it
        res["exception"] = f"{type(e).__name__}: {e}"
    with open(res_path, "w") as f:
        json.dump(res, f)
    return 0


def child_exec(build_dir, timeout_s=90):
    n = [0]

    def run(req):
        n[0] += 1
        rq, rs = os.path.join(build_dir, f"req-{n[0]:03d}.json"), os.path.join(build_dir, f"res-{n[0]:03d}.json")
        with open(rq, "w") as f:
            json.dump(req, f)
        if os.path.exists(rs):
            os.remove(rs)
        try:
            cp = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", rq, rs], capture_output=True,
                                text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"rc": -9, "exception": f"the child process did not finish in {timeout_s} s (killed)"}
        if not os.path.exists(rs):
            tail = " | ".join((cp.stderr or cp.stdout or "").strip().splitlines()[-3:])
            return {"rc": -8, "exception": f"the child process died with code {cp.returncode}: {tail[:400]}"}
        with open(rs) as f:
            return json.load(f)

    return run


# ------------------------------------------------------------------------------------------------ compile
def find_nvcc(override=None):
    cands = [override] if override else []
    w = shutil.which("nvcc")
    cands += [w] if w else []
    cands += ["/usr/local/cuda/bin/nvcc"]
    for pat in ("/usr/local/lib/python3*/dist-packages/nvidia/cu*/bin/nvcc",
                "/usr/local/lib/python3*/site-packages/nvidia/cu*/bin/nvcc", "/usr/local/cuda-*/bin/nvcc"):
        cands += sorted(glob.glob(pat))
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c, cands
    return None, cands


def nvcc_version(nvcc):
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60).stdout
        return next((ln.strip() for ln in out.splitlines() if "release" in ln), out.strip()[-80:])
    except Exception as e:  # noqa: BLE001
        return f"version unknown ({type(e).__name__})"


def enum_report(nvcc):
    """Print the tensor-map data-type lines of the cuda.h this nvcc will include. Returns True if both extra names
    (ALIGN8B, 16U6) are there, False if the header was found without them, None if no header was found."""
    cands = [os.path.join(os.path.dirname(os.path.realpath(nvcc)), "..", "include", "cuda.h"),
             os.path.join(os.path.dirname(nvcc), "..", "include", "cuda.h"), "/usr/local/cuda/include/cuda.h"]
    for h in cands:
        if os.path.isfile(h):
            seen = {}
            with open(h, errors="replace") as f:
                for i, line in enumerate(f, 1):
                    for name in ("16U4_ALIGN8B", "16U4_ALIGN16B", "16U6_ALIGN16B"):
                        if "CU_TENSOR_MAP_DATA_TYPE_" + name in line and name not in seen:
                            seen[name] = i
            log(f"cuda.h: {os.path.normpath(h)}: " + ", ".join(f"{k} at line {v}" for k, v in seen.items()) +
                ("" if len(seen) == 3 else f"; MISSING {sorted(set(('16U4_ALIGN8B', '16U4_ALIGN16B', '16U6_ALIGN16B')) - set(seen))}"))
            return "16U4_ALIGN8B" in seen and "16U6_ALIGN16B" in seen
    log("cuda.h: not found next to nvcc (the compiler will find its own); the extra enum names are tried blind")
    return None


def lib_dirs(nvcc):
    """Directories that hold the static CUDA runtime, for the case where nvcc's own link line misses them."""
    roots = [os.path.join(os.path.dirname(os.path.realpath(nvcc)), ".."), os.path.join(os.path.dirname(nvcc), ".."),
             "/usr/local/cuda"] + sorted(glob.glob("/usr/local/lib/python3*/*-packages/nvidia/cu*"))
    out = []
    for r in roots:
        for sub in ("lib64", "lib", os.path.join("targets", "sbsa-linux", "lib"),
                    os.path.join("targets", "aarch64-linux", "lib")):
            d = os.path.normpath(os.path.join(r, sub))
            if d not in out and os.path.isfile(os.path.join(d, "libcudart_static.a")):
                out.append(d)
    return out


def compile_form(nvcc, arch, form, build_dir, extra_enums):
    out = os.path.join(build_dir, f"k7e2_f{form}.so")
    base = [nvcc] + NVCC_FLAGS + [f"-gencode=arch=compute_{arch},code=sm_{arch}", f"-DK7_CP_FORM={form}",
                                  f"-DK7_EXTRA_ENUMS={int(extra_enums)}", "-I", HERE, KERNEL_CU, "-o", out]
    text, ok, tried = "", False, []
    for extra in [[]] + [["-L", d] for d in lib_dirs(nvcc)]:
        cmd = base + extra
        tried.append(" ".join(cmd))
        if os.path.exists(out):
            os.remove(out)
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=170, cwd=build_dir)
            text = (cp.stdout or "") + (cp.stderr or "")
            ok = cp.returncode == 0 and os.path.isfile(out)
        except subprocess.TimeoutExpired:
            text, ok = "nvcc did not finish in 170 s", False
        except OSError as e:
            text, ok = f"{type(e).__name__}: {e}", False
        if ok or "cannot find -l" not in text:      # only a missing runtime library is worth another -L
            break
    with open(os.path.join(build_dir, f"k7e2_f{form}.log"), "w") as f:
        f.write("\n".join(tried) + "\n" + text)
    return ok, out, text


def error_lines(text, n=4):
    """The lines of a compiler log that say what went wrong (not the warnings, not the source echoes)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hits = [ln for ln in lines if "warning" not in ln.lower() and
            any(k in ln for k in (" error", "error:", "fatal", "cannot find", "undefined reference", "did not finish"))]
    return (hits or lines[-n:])[:n]


def build_all(nvcc, arch, build_dir):
    """Compile the control form, then the forms under test. Returns {form: path} or None if the control fails."""
    have = enum_report(nvcc)
    extra = have is not False
    log("flags: nvcc " + " ".join(NVCC_FLAGS) + f" -gencode=arch=compute_{arch},code=sm_{arch} -DK7_CP_FORM=<n>"
        " (kernels/w8-persist/build.sh:10-12)")
    libs = {}
    t0 = time.time()
    ok, path, text = compile_form(nvcc, arch, 0, build_dir, extra)
    if not ok and extra and ("ALIGN8B" in text or "16U6" in text):
        log("form 0: the extra enum names are not in this cuda.h; compiling without them (the ALIGN8B and 6-bit "
            "TMA cases will report 'refused')")
        extra = False
        ok, path, text = compile_form(nvcc, arch, 0, build_dir, extra)
    if not ok:
        log("could not compile the control kernel (form 0, no new instruction in it):")
        for ln in error_lines(text, 8):
            log("  " + ln[:300])
        return None
    libs[0] = path
    log(f"form 0 compiled: {FORM_STR[0]}")
    for form in (1, 2):
        ok, path, text = compile_form(nvcc, arch, form, build_dir, extra)
        if ok:
            libs[form] = path
            log(f"form {form} compiled: {FORM_STR[form]}")
        else:
            log(f"form {form} DID NOT COMPILE: {FORM_STR[form]}")
            for ln in error_lines(text):
                log("  " + ln[:300])
    fallbacks = ([3] if 1 not in libs else []) + ([4] if 2 not in libs else [])
    for form in fallbacks + ([5] if fallbacks else []):
        ok, path, text = compile_form(nvcc, arch, form, build_dir, extra)
        if ok:
            libs[form] = path
            log(f"form {form} compiled (fallback shape): {FORM_STR[form]}")
        else:
            log(f"form {form} DID NOT COMPILE: {FORM_STR[form]}")
            for ln in error_lines(text):
                log("  " + ln[:300])
    log(f"compiled {sorted(libs)} in {time.time() - t0:.0f} s; logs in {build_dir}")
    return libs


def main(argv=None):
    ap = argparse.ArgumentParser(description="K7 E2: a packed scale column in shared and tensor memory")
    ap.add_argument("--build-dir", default=os.path.join(HERE, "k7-e2-build"), help="writable: .so files, logs")
    ap.add_argument("--json", default=None, help="write every raw result here")
    ap.add_argument("--nvcc", default=None, help="compiler to use (default: PATH, then the usual places)")
    ap.add_argument("--arch", default="103a", help="gencode target suffix (GB300: 103a)")
    ap.add_argument("--skip-mma", action="store_true", help="skip stage 3 (the multiply)")
    ap.add_argument("--child", nargs=2, metavar=("REQ", "RES"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return child_main(*a.child)
    a.build_dir = os.path.abspath(a.build_dir)
    os.makedirs(a.build_dir, exist_ok=True)
    log("K7 E2: packed scale column -> TMA load -> shared memory -> tcgen05.cp -> tensor memory")
    for p in (KERNEL_CU, PTX_CUH):
        if not os.path.isfile(p):
            log(f"could not compile: missing source file {p}")
            return 2
    nvcc, cands = find_nvcc(a.nvcc)
    if nvcc is None:
        log(f"could not compile: no nvcc found (looked at {[c for c in cands if c]})")
        return 2
    if a.nvcc and nvcc != a.nvcc:
        log(f"--nvcc {a.nvcc} is not an executable file: using the one found instead")
    log(f"compiler: {nvcc} ({nvcc_version(nvcc)})")
    libs = build_all(nvcc, a.arch, a.build_dir)
    if libs is None:
        return 2
    ctx = Ctx(child_exec(a.build_dir), libs, skip_mma=a.skip_mma)
    S = experiment(ctx)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"schema": "k7-e2/1", "decision": S.get("decision"), "why": S.get("why"),
                       "verdicts": ctx.verdicts, "state": {k: v for k, v in S.items() if k not in ("why",)},
                       "cases": ctx.cases, "forms": {str(k): FORM_STR[k] for k in libs}}, f, indent=1, default=str)
        log(f"wrote {a.json}")
    return int(S["exit_code"])


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - one line for the runner, then the trace
        print(f"{TAG} could not run: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(2)
