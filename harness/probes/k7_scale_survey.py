#!/usr/bin/env python3
"""K7 survey: what can be won from the MXFP4 block scales (E8M0, one byte per 32 weights) of the routed experts.

CPU only. Reads the safetensors shards by header offsets, one layer at a time (under 3 GiB of RAM), and prints a
short table. Run on the Station:

    uv run --with numpy python k7_scale_survey.py /raid/models/DeepSeek-V4.1-Flash-1017 --quick
    uv run --with numpy python k7_scale_survey.py /raid/models/DeepSeek-V4.1-Flash-1017 \
        --rowmap /raid/work/gt/gracetier/tree/fastpath/hook/rowmap-allres.json --json /raid/work/logs/k7-survey.json

--quick reads layers 0,10,20,30,39 (2.0 GiB of scales). The default reads all 40 layers (15.8 GiB of scales, about
5 to 10 minutes: an estimate, not measured). Partial results are written to --json after every layer.

What it measures, and which design option each number decides (research/K7-packed-scales.md):
  A. code range and distinct values over EVERY scale read (not a sample of matrices). The 4-bit pack of option (a)
     needs every code inside 0x70..0x7F (then the low nibble is the whole code), or at least max - min <= 15.
  B. entropy per code (order 0, given the left neighbour, given the row above, per 4-code word): the bound for any
     packing denser than 4 bits, and what a 256-entry word dictionary would cover.
  C. page de-duplication (option b): the scale banks are laid out exactly as MegaMoE reads them (int32 words of 4
     codes, L1 rows gate/up interleaved by 8, the UTCCP 4x32 row transpose, MN-major storage [group][K word][row],
     groups in --rowmap order or in expert order) and cut at the CUDA VMM granule (2 MiB) from each bank's base. It
     counts byte-identical granules inside a bank, inside a layer and across the model, and the same at 64 KiB and
     4 KiB (smaller units do not exist on this driver; they show how far from useful the 2 MiB number is).
  D. whole-expert duplicates: experts whose complete L1 or L2 scale block is byte-identical to another expert's.

The bank layout is a port of runtime/dg_mega.py (mega_l1_sf, mega_l2_sf: sf_words, interleave_rows, utccp_rows,
to_major "mn"); t_k7_survey.py checks the port against that file with torch.
"""
import argparse
import glob
import json
import math
import os
import re
import struct
import sys
import time

import numpy as np

NAME = re.compile(r"^(?:model\.)?layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(?:scale|weight_scale_inv)$")
GRAN = 8          # gate/up interleave granularity (deep_gemm mega/__init__.py _interleave_weights)
UTCCP = 128       # rows per UTCCP transpose group (4 x 32)
PAGES = (2 << 20, 64 << 10, 4 << 10)
QUICK = (0, 10, 20, 30, 39)
PAIR_GROUPS = 48  # groups per layer used for the neighbour and word statistics (part B); part A reads every code
GiB = 1024.0 ** 3
HOT_GIB = 0.70    # GPU memory per extra hot expert per layer (PLAN.md; 0.659 GiB of weights by the shapes)


# ------------------------------------------------------------------------------------------------ checkpoint index
def scan(root):
    """{(layer, expert, proj): (file, offset, nbytes, shape)} for every routed-expert scale tensor."""
    idx = {}
    for f in sorted(glob.glob(root.rstrip("/") + "/*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for name, t in hdr.items():
            m = NAME.match(name)
            if m is None:
                continue
            a, b = t["data_offsets"]
            idx[(int(m.group(1)), int(m.group(2)), m.group(3))] = (f, base + a, b - a, tuple(t["shape"]))
    return idx


class Reader:
    def __init__(self):
        self.fh = {}

    def read(self, entry, out):
        f, off, size, shape = entry
        if size != out.size:
            raise ValueError(f"{f}@{off}: {size} bytes on disk, {out.size} expected (shape {shape})")
        fh = self.fh.get(f)
        if fh is None:
            fh = self.fh[f] = open(f, "rb", buffering=0)
        fh.seek(off)
        got = fh.readinto(memoryview(out).cast("B"))
        if got != size:
            raise IOError(f"{f}@{off}: short read {got} of {size}")


# ------------------------------------------------------------------------------------------------ bank layout (pure)
def interleave_rows(w):
    """[G, 2I, C] rows [gate | up] -> [gate 0..7, up 0..7, gate 8..15, ...] (dg_mega.interleave_rows)."""
    g, n, c = w.shape
    half = n // 2
    gate = w[:, :half].reshape(g, half // GRAN, GRAN, c)
    up = w[:, half:].reshape(g, half // GRAN, GRAN, c)
    return np.stack([gate, up], axis=2).reshape(g, n, c)


def utccp_rows(w):
    """Rows permuted inside each 128-row group: old a*128 + i*32 + j -> new a*128 + j*4 + i (dg_mega.utccp_rows)."""
    g, n, c = w.shape
    return w.reshape(g, n // UTCCP, 4, 32, c).transpose(0, 1, 3, 2, 4).reshape(g, n, c)


def bank_l1(c1, c3):
    """w1 / w3 codes uint8 [G, I, K/32] -> the L1 bank's bytes in storage order: int32 [G, K/128, 2I]."""
    w = np.concatenate([np.ascontiguousarray(c1).view("<i4"), np.ascontiguousarray(c3).view("<i4")], axis=1)
    return np.ascontiguousarray(utccp_rows(interleave_rows(w)).transpose(0, 2, 1))


def bank_l2(c2):
    """w2 codes uint8 [G, D, K/32] -> the L2 bank's bytes in storage order: int32 [G, K/128, D]."""
    return np.ascontiguousarray(utccp_rows(np.ascontiguousarray(c2).view("<i4")).transpose(0, 2, 1))


def pack_nibbles(u8):
    """Dense code bytes [..., 2m] (every code in 0x70..0x7F) -> packed [..., m]: byte j = code 2j's low nibble in
    bits 0..3, code 2j+1's in bits 4..7. The same function as runtime/gracetier_sfpack.pack_bytes of the K7 patch."""
    return (u8[..., 0::2] & 15) | ((u8[..., 1::2] & 15) << 4)


def unpack_nibbles(p):
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = (p & 15) | 0x70
    out[..., 1::2] = (p >> 4) | 0x70
    return out


# ------------------------------------------------------------------------------------------------ hashing
_MULT = {}


def _mult(nwords):
    m = _MULT.get(nwords)
    if m is None:
        rng = np.random.default_rng(0x4B37)                       # fixed: hashes compare across layers and runs
        m = _MULT[nwords] = rng.integers(1, 2 ** 63, size=nwords, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    return m


def chunk_hashes(buf, nbytes):
    """64-bit hashes of the full nbytes chunks of a byte buffer (wrapping sum of the uint64 words times fixed odd
    multipliers plus a second mix; position-sensitive). Returns (hashes, constant) where constant[i] says the chunk
    is one repeated 8-byte word."""
    words = buf.view(np.uint64)
    cw = nbytes // 8
    n = words.size // cw
    if n == 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    v = words[:n * cw].reshape(n, cw)
    m = _mult(cw)
    h = np.zeros(n, np.uint64)
    const = np.ones(n, bool)
    step = max(1, (64 << 20) // nbytes)                           # 64 MiB of temporaries at a time
    for i in range(0, n, step):
        blk = v[i:i + step]
        with np.errstate(over="ignore"):
            a = (blk * m).sum(axis=1, dtype=np.uint64)
            b = (blk ^ (blk >> np.uint64(29))).sum(axis=1, dtype=np.uint64)
            h[i:i + step] = a ^ (b * np.uint64(0x9E3779B97F4A7C15))
        const[i:i + step] = (blk == blk[:, :1]).all(axis=1)
    return h, const


def entropy(counts):
    c = np.asarray(counts, dtype=np.float64).ravel()
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return max(0.0, float(-(p * np.log2(p)).sum()))


def cond_entropy(joint):
    """H(second | first) from a [256, 256] count table."""
    tot = joint.sum()
    if tot == 0:
        return 0.0
    rows = joint.sum(axis=1)
    return float(sum(rows[a] / tot * entropy(joint[a]) for a in np.nonzero(rows)[0]))


# ------------------------------------------------------------------------------------------------ the survey
class Acc:
    def __init__(self, pages):
        self.hist = {p: np.zeros(256, np.int64) for p in ("w1", "w3", "w2")}
        self.left = np.zeros((256, 256), np.int64)
        self.up = np.zeros((256, 256), np.int64)
        self.words = np.zeros(65536, np.int64)
        self.words_ok = True
        self.rows = 0
        self.rows_const = 0
        self.bytes = 0
        self.layer_rng = {}
        self.pages = pages
        self.ph = {p: {"l1": [], "l2": []} for p in pages}       # per layer arrays of hashes
        self.pconst = {p: 0 for p in pages}
        self.eh = {"l1": [], "l2": []}
        self.tail = {p: 0 for p in pages}                         # bytes after the last full chunk (not hashed)
        self.packcheck = {"bytes": 0, "mismatch": 0, "skipped_layers": 0}

    def codes(self, proj, c):
        """Part A on every code of c [G, N, K/32]; the neighbour and word tables of part B on the first PAIR_GROUPS
        groups (they only feed entropy estimates)."""
        flat = c.reshape(-1)
        self.hist[proj] += np.bincount(flat, minlength=256)
        self.bytes += flat.size
        self.rows += c.shape[0] * c.shape[1]
        self.rows_const += int((c == c[..., :1]).all(axis=-1).sum())
        s = c[:PAIR_GROUPS]
        a = s[..., :-1].astype(np.uint16) << 8
        self.left += np.bincount((a | s[..., 1:]).reshape(-1), minlength=65536).reshape(256, 256)
        a = s[:, :-1, :].astype(np.uint16) << 8
        self.up += np.bincount((a | s[:, 1:, :]).reshape(-1), minlength=65536).reshape(256, 256)
        if self.words_ok and int(s.min()) >= 0x70 and int(s.max()) <= 0x7F:
            n = (s & 15).astype(np.uint16).reshape(-1, 4)
            key = n[:, 0] | (n[:, 1] << 4) | (n[:, 2] << 8) | (n[:, 3] << 12)
            self.words += np.bincount(key, minlength=65536)
        else:
            self.words_ok = False


def survey_layer(acc, idx, rd, layer, experts, order):
    """experts: the expert ids to read; order: the same ids in bank group order (hot then cold, or ascending)."""
    e0 = experts[0]
    I, k1 = idx[(layer, e0, "w1")][3]
    D, k2 = idx[(layer, e0, "w2")][3]
    pos = {e: i for i, e in enumerate(order)}
    G = len(order)
    c = {"w1": np.empty((G, I, k1), np.uint8), "w3": np.empty((G, I, k1), np.uint8),
         "w2": np.empty((G, D, k2), np.uint8)}
    for e in experts:
        for p in ("w1", "w3", "w2"):
            ent = idx[(layer, e, p)]
            if ent[3] != c[p].shape[1:]:
                raise ValueError(f"layer {layer} expert {e} {p}: shape {ent[3]}, expected {c[p].shape[1:]}")
            rd.read(ent, c[p][pos[e]])
    lo = min(int(v.min()) for v in c.values())
    hi = max(int(v.max()) for v in c.values())
    acc.layer_rng[layer] = (lo, hi)
    for p in ("w1", "w3", "w2"):
        acc.codes(p, c[p])
    if not acc.pages:
        return
    banks = {"l1": bank_l1(c["w1"], c["w3"]), "l2": bank_l2(c["w2"])}
    del c
    for kind, b in banks.items():
        raw = b.reshape(-1).view(np.uint8)
        for pg in acc.pages:
            h, const = chunk_hashes(raw, pg)
            acc.ph[pg][kind].append(h)
            acc.pconst[pg] += int(const.sum())
            acc.tail[pg] += raw.size - h.size * pg
        per = b.reshape(G, -1).view(np.uint8)
        acc.eh[kind].append(chunk_hashes(per.reshape(-1), per.shape[1])[0])
        if lo >= 0x70 and hi <= 0x7F:                              # the pack of option (a), checked on real bytes
            sub = per[: min(G, 8)]
            acc.packcheck["bytes"] += sub.size
            acc.packcheck["mismatch"] += int((unpack_nibbles(pack_nibbles(sub)) != sub).sum())
        elif kind == "l1":
            acc.packcheck["skipped_layers"] += 1


def dup_stats(arrs):
    """(chunks, distinct, duplicates inside one array summed, duplicates over all arrays) for a list of hash arrays."""
    if not arrs:
        return 0, 0, 0, 0
    allh = np.concatenate(arrs)
    inside = sum(int(a.size - np.unique(a).size) for a in arrs)
    distinct = int(np.unique(allh).size)
    return int(allh.size), distinct, inside, int(allh.size - distinct)


def report(acc, meta):
    out = dict(meta)
    hist = sum(acc.hist.values())
    used = np.nonzero(hist)[0]
    total = int(hist.sum())
    out["codes_read"] = total
    out["distinct"] = [int(x) for x in used]
    out["min"], out["max"] = (int(used.min()), int(used.max())) if used.size else (None, None)
    out["outside_0x70_0x7F"] = int(hist[:0x70].sum() + hist[0x80:].sum())
    out["layers_span_le_15"] = all(hi - lo <= 15 for lo, hi in acc.layer_rng.values())
    out["layer_ranges"] = {str(k): list(v) for k, v in sorted(acc.layer_rng.items())}
    out["share_by_code"] = {str(int(x)): round(float(hist[x]) / total, 6) for x in used} if total else {}
    out["H0"] = entropy(hist)
    out["H0_by_proj"] = {p: entropy(h) for p, h in acc.hist.items()}
    out["H_left"] = cond_entropy(acc.left)
    out["H_up"] = cond_entropy(acc.up)
    out["rows_const_share"] = acc.rows_const / acc.rows if acc.rows else None
    if acc.words_ok and acc.words.sum():
        w = np.sort(acc.words)[::-1]
        tot = float(w.sum())
        out["word_H_per_code"] = entropy(acc.words) / 4
        out["word_distinct"] = int((acc.words > 0).sum())
        out["word_top16"] = float(w[:16].sum() / tot)
        out["word_top256"] = float(w[:256].sum() / tot)
    else:
        out["word_H_per_code"] = out["word_distinct"] = out["word_top16"] = out["word_top256"] = None
    out["pages"] = {}
    for pg in acc.pages:
        row = {}
        both = [a for k in ("l1", "l2") for a in acc.ph[pg][k]]
        n, distinct, inside, over_all = dup_stats(both)
        per_layer = 0
        for i in range(len(acc.ph[pg]["l1"])):
            lay = np.concatenate([acc.ph[pg]["l1"][i], acc.ph[pg]["l2"][i]])
            per_layer += int(lay.size - np.unique(lay).size)
        row.update(chunks=n, distinct=distinct, dup_inside_bank=inside, dup_inside_layer=per_layer,
                   dup_model=over_all, constant=acc.pconst[pg], bytes_not_hashed=acc.tail[pg],
                   dup_share=(over_all / n if n else None), dup_bytes=over_all * pg)
        out["pages"][str(pg)] = row
    out["expert_dups"] = {}
    for kind in ("l1", "l2"):
        n, distinct, inside, over_all = dup_stats(acc.eh[kind])
        out["expert_dups"][kind] = {"blocks": n, "distinct": distinct, "dup_inside_layer": inside,
                                    "dup_model": over_all}
    out["packcheck"] = acc.packcheck
    return out


def table(r):
    L = []
    frac = r["codes_read"] / r["codes_model"] if r["codes_model"] else 0.0
    L.append(f"K7 scale survey: {r['root']}")
    L.append(f"  read {r['codes_read'] / GiB:.2f} GiB of scales = {100 * frac:.1f} % of the routed experts "
             f"({len(r['layers'])} layers x {r['experts_per_layer']} experts) in {r['seconds']:.0f} s; group order: "
             f"{r['order']}")
    L.append("A. code range (decides the 4-bit pack of option a)")
    L.append(f"  distinct values {len(r['distinct'])}: {r['distinct']}")
    L.append(f"  codes outside 0x70..0x7F: {r['outside_0x70_0x7F']}  -> low-nibble pack "
             f"{'VALID' if r['outside_0x70_0x7F'] == 0 else 'NOT valid'}; every layer spans <= 15: "
             f"{'yes' if r['layers_span_le_15'] else 'NO'}")
    top = sorted(r["share_by_code"].items(), key=lambda kv: -kv[1])[:5]
    L.append("  most common: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in top))
    L.append("B. entropy, bits per 8-bit code (bounds for anything denser than 4 bits)")
    L.append(f"  order 0 {r['H0']:.3f} | given left neighbour {r['H_left']:.3f} | given row above {r['H_up']:.3f} | "
             f"per 4-code word " + (f"{r['word_H_per_code']:.3f}" if r["word_H_per_code"] is not None else "n/a"))
    if r["word_distinct"] is not None:
        L.append(f"  4-code words: {r['word_distinct']} distinct; top 16 cover {100 * r['word_top16']:.2f} %, top 256 "
                 f"cover {100 * r['word_top256']:.2f} % (a 1-byte word index needs 100 % or an escape)")
    L.append(f"  rows whose scales are all one value: {100 * (r['rows_const_share'] or 0):.2f} %")
    nd = len(r["distinct"])
    fixed = [("4-bit low nibble", 4.0, r["outside_0x70_0x7F"] == 0),
             ("5 codes in 16 bits (base 9)", 3.2, nd <= 9)]
    ib = max(1, math.ceil(math.log2(max(nd, 2))))
    if ib < 4:
        fixed.append((f"{ib}-bit index of {nd} values", float(ib), True))
    sf_all = r["sf_gib_all_groups"]
    L.append(f"  fixed-rate packs against the {sf_all:.2f} GiB of routed-expert scales in this checkpoint:")
    for name, bits, ok in fixed:
        freed = sf_all * (1 - bits / 8)
        L.append(f"    {name:<34s} {bits:.1f} bits  frees {freed:5.2f} GiB = +{freed / HOT_GIB:4.1f} hot/layer  "
                 f"{'valid' if ok else 'NOT valid on this data'}")
    L.append(f"    entropy bound (order 0)            {r['H0']:.1f} bits  frees {sf_all * (1 - r['H0'] / 8):5.2f} GiB"
             f" = +{sf_all * (1 - r['H0'] / 8) / HOT_GIB:4.1f} hot/layer  (not a fixed-rate format)")
    if r["pages"]:
        L.append("C. page de-duplication (option b); banks in MegaMoE's layout, cut from each bank's base")
        L.append("  unit        chunks   distinct  dup in bank  dup in layer  dup in model   constant   dup share")
        for pg, row in r["pages"].items():
            unit = f"{int(pg) >> 20} MiB" if int(pg) >= 1 << 20 else f"{int(pg) >> 10} KiB"
            L.append(f"  {unit:>7s} {row['chunks']:>10d} {row['distinct']:>10d} {row['dup_inside_bank']:>12d} "
                     f"{row['dup_inside_layer']:>13d} {row['dup_model']:>13d} {row['constant']:>10d}   "
                     f"{100 * (row['dup_share'] or 0):7.3f} %")
        p2 = r["pages"].get(str(2 << 20))
        if p2:
            scale = 1.0 / frac if frac else 0.0
            L.append(f"  2 MiB is the only unit CUDA VMM can map on this driver: {p2['dup_model']} duplicate granules "
                     f"= {p2['dup_bytes'] / GiB:.3f} GiB in what was read (about {p2['dup_bytes'] * scale / GiB:.2f} "
                     f"GiB if the rest is alike) = +{p2['dup_bytes'] * scale / GiB / HOT_GIB:.1f} hot/layer")
        L.append("D. whole-expert duplicates (same scale block as another expert)")
        for kind, row in r["expert_dups"].items():
            L.append(f"  {kind}: {row['dup_model']} of {row['blocks']} blocks duplicate another "
                     f"({row['dup_inside_layer']} inside their own layer)")
        pc = r["packcheck"]
        L.append(f"E. pack self-check on bank bytes: {pc['bytes']} bytes packed and unpacked, {pc['mismatch']} "
                 f"mismatches" + (f", {pc['skipped_layers']} layers skipped (codes out of range)"
                                  if pc["skipped_layers"] else ""))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--layers", default="all", help="all, or a comma list")
    ap.add_argument("--quick", action="store_true", help=f"layers {','.join(map(str, QUICK))} only")
    ap.add_argument("--experts", default="all", help="all, or the first N experts of each layer")
    ap.add_argument("--rowmap", default=None, help="rowmap-allres.json: bank group order = hot then cold")
    ap.add_argument("--no-pages", action="store_true", help="skip parts C, D, E (halves the run time)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    idx = scan(a.root)
    if not idx:
        print(f"no routed-expert scale tensors under {a.root}", file=sys.stderr)
        return 2
    layers_all = sorted({k[0] for k in idx})
    if a.quick:
        layers = [l for l in QUICK if l in layers_all]
    elif a.layers == "all":
        layers = layers_all
    else:
        layers = [int(x) for x in a.layers.split(",")]
    rowmap = None
    if a.rowmap:
        with open(a.rowmap) as f:
            doc = json.load(f)
        rowmap = doc.get("layers", doc)
    acc = Acc(() if a.no_pages else PAGES)
    rd = Reader()
    n_exp = None
    model_bytes = sum(v[2] for v in idx.values())
    meta = {"root": a.root, "layers": layers, "codes_model": model_bytes,
            "order": "rowmap (hot then cold)" if rowmap else "expert id",
            "sf_gib_all_groups": model_bytes / GiB}
    for n, layer in enumerate(layers):
        ids = sorted(k[1] for k in idx if k[0] == layer and k[2] == "w1")
        if a.experts != "all":
            ids = ids[: int(a.experts)]
        order = ids
        if rowmap is not None:
            v = rowmap[str(layer)]
            order = [int(e) for e in list(v["hot"]) + list(v["cold"]) if int(e) in set(ids)]
            if sorted(order) != ids:
                raise ValueError(f"rowmap layer {layer} does not cover the experts read")
        n_exp = len(ids)
        survey_layer(acc, idx, rd, layer, ids, order)
        meta.update(experts_per_layer=n_exp, seconds=time.time() - t0, layers=layers[: n + 1])
        if a.json:
            with open(a.json + ".tmp", "w") as f:
                json.dump(report(acc, meta), f)
            os.replace(a.json + ".tmp", a.json)
        lo, hi = acc.layer_rng[layer]
        print(f"  layer {layer:2d}: {n_exp} experts, codes {lo}..{hi}, {time.time() - t0:.0f} s", file=sys.stderr,
              flush=True)
    meta["layers"] = layers
    meta["seconds"] = time.time() - t0
    r = report(acc, meta)
    print(table(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
