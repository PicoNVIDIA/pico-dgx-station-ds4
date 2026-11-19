#!/usr/bin/env python3
"""Draws the banner and the charts from results/summary.json in the colors of assets/theme.json.

    python3 tools/make_assets.py

Writes assets/banner.svg, assets/chart-one-user.svg, assets/chart-32-users.svg, assets/chart-scaling.svg and
assets/preview.html (the README's visuals on one page, to judge the look without pushing).
Standard library only. To change how the repo looks, edit assets/theme.json and run this again.
"""
import html
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
T = json.loads((ROOT / "assets" / "theme.json").read_text())
S = json.loads((ROOT / "results" / "summary.json").read_text())
LABEL = {s["id"]: s["label"] for s in S["systems"]}
CELLS = S["cells"]
FONT = T["font"]


def esc(s):
    return html.escape(str(s), quote=True)


def num(v):
    return f"{v:,.1f}" if isinstance(v, float) and v < 1000 and v != int(v) else f"{int(round(v)):,}"


def lohi(v):
    return (v[0], v[1]) if isinstance(v, list) else (v, v)


def vtext(v):
    lo, hi = lohi(v)
    return num(lo) if lo == hi else f"{num(lo)} to {num(hi)}"


def text(x, y, s, size=16, fill=None, weight=400, anchor="start", spacing=0):
    sp = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill or T["text"]}" text-anchor="{anchor}"{sp}>{esc(s)}</text>')


def frame(w, h, body):
    r = T["radius"]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">\n'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="{r}" fill="{T["panel"]}" stroke="{T["panel_border"]}"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def banner():
    b = T["banner"]
    w, h, r = 1280, 360, T["radius"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">',
           f'<defs><clipPath id="c"><rect width="{w}" height="{h}" rx="{r}"/></clipPath>'
           f'<linearGradient id="g" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{T["accent_dim"]}"/>'
           f'<stop offset="1" stop-color="{T["accent"]}"/></linearGradient></defs>',
           f'<g clip-path="url(#c)"><rect width="{w}" height="{h}" fill="{T["bg"]}"/>']
    # slanted bars, top right: the only decoration
    for i in range(7):
        x = 900 + i * 62
        op = 0.18 + 0.12 * i
        out.append(f'<polygon points="{x},0 {x + 34},0 {x - 86},210 {x - 120},210" fill="url(#g)" opacity="{op:.2f}"/>')
    out.append(f'<rect x="0" y="0" width="10" height="{h}" fill="{T["accent"]}"/>')
    out.append(text(64, 74, b["kicker"], 17, T["muted"] if T["accent"].lower() == T["text"].lower() else T["accent"], 700, spacing=3))
    out.append(text(62, 152, b["title"], 72, T["text"], 800))
    out.append(text(64, 196, b["subtitle"], 25, T["muted"], 400))
    tiles = b["tiles"]
    gap, x0, y0, th = 20, 64, 238, 88
    tw = (w - 2 * x0 - gap * (len(tiles) - 1)) / len(tiles)
    for i, t in enumerate(tiles):
        x = x0 + i * (tw + gap)
        out.append(f'<rect x="{x:.1f}" y="{y0}" width="{tw:.1f}" height="{th}" rx="10" fill="{T["panel"]}" stroke="{T["panel_border"]}"/>')
        out.append(text(x + 24, y0 + 60, t["big"], 50, T["accent"], 800))
        out.append(text(x + tw - 24, y0 + 53, t["small"], 20, T["muted"], 400, "end"))
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


def hbar(title, sub, cell, ids, note=None):
    rows = [(i, CELLS[cell][i]) for i in ids if CELLS[cell].get(i) is not None]
    w, top, rh = 960, 104, 54
    h = top + rh * len(rows) + (58 if note else 30)
    x0, x1 = 300, 760
    vmax = max(lohi(v)[1] for _, v in rows)
    out = [text(36, 48, title, 26, T["text"], 700), text(36, 76, sub, 16, T["muted"])]
    for k, (sid, v) in enumerate(rows):
        y = top + k * rh
        lo, hi = lohi(v)
        col = T["series"].get(sid, T["muted"])
        ours = sid.startswith("ours")
        out.append(text(x0 - 18, y + 27, LABEL[sid], 16, T["text"] if ours else T["muted"], 700 if ours else 400, "end"))
        out.append(f'<rect x="{x0}" y="{y + 8}" width="{x1 - x0}" height="28" rx="5" fill="{T["grid"]}" opacity="0.55"/>')
        wl = (x1 - x0) * lo / vmax
        wh = (x1 - x0) * hi / vmax
        if hi > lo:
            out.append(f'<rect x="{x0}" y="{y + 8}" width="{wh:.1f}" height="28" rx="5" fill="{col}" opacity="0.45"/>')
        out.append(f'<rect x="{x0}" y="{y + 8}" width="{wl:.1f}" height="28" rx="5" fill="{col}"/>')
        out.append(text(x1 + 16, y + 28, vtext(v), 18, T["accent"] if ours else T["text"], 700))
    if note:
        out.append(text(36, h - 22, note, 14, T["muted"]))
    return frame(w, h, out)


def scaling():
    ids = ["stock_vllm", "sglang_patched", "public_recipe", "ours_1m", "ours_262k"]
    xs = [("sampled_1user", "1"), ("users_2", "2"), ("users_4", "4"), ("users_8", "8"), ("users_16", "16"), ("users_32", "32")]
    w, h = 960, 560
    px0, px1, py0, py1 = 86, 916, 172, 478
    ymax = 2500
    X = lambda k: px0 + (px1 - px0) * k / (len(xs) - 1)
    Y = lambda v: py1 - (py1 - py0) * v / ymax
    out = [text(36, 48, "Throughput by number of users", 26, T["text"], 700),
           text(36, 76, "Output tokens per second, all users together. GSM8K prompts, sampled, 60 s windows.", 16, T["muted"])]
    lx, ly = 36, 100
    for sid in ids:
        col = T["series"][sid]
        need = 30 + 8.8 * len(LABEL[sid])
        if lx + need > w - 36:
            lx, ly = 36, ly + 24
        out.append(f'<rect x="{lx}" y="{ly}" width="22" height="6" rx="3" fill="{col}"/>')
        out.append(text(lx + 30, ly + 9, LABEL[sid], 14, T["text"] if sid.startswith("ours") else T["muted"]))
        lx += need + 26
    for v in range(0, ymax + 1, 500):
        out.append(f'<line x1="{px0}" y1="{Y(v):.1f}" x2="{px1}" y2="{Y(v):.1f}" stroke="{T["grid"]}"/>')
        out.append(text(px0 - 12, Y(v) + 5, num(v), 14, T["muted"], 400, "end"))
    for k, (_, lab) in enumerate(xs):
        out.append(text(X(k), py1 + 28, lab, 15, T["muted"], 400, "middle"))
    out.append(text((px0 + px1) / 2, py1 + 58, "users at once", 15, T["muted"], 400, "middle"))
    for sid in ids:
        col = T["series"][sid]
        pts = [(X(k), Y(lohi(CELLS[c][sid])[0])) for k, (c, _) in enumerate(xs) if CELLS[c].get(sid) is not None]
        ours = sid.startswith("ours")
        dash = "" if len(pts) == len(xs) else ' stroke-dasharray="7 7"'
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="{4 if ours else 2.5}" stroke-linejoin="round"{dash} '
                   f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5.5 if ours else 4}" fill="{col}" stroke="{T["panel"]}" stroke-width="2"/>')
    return frame(w, h, out)


def memory_diagram():
    w, h = 960, 404
    A, D, M = T["accent"], T["accent_dim"], T["muted"]
    out = [text(36, 48, "Where the model lives", 26, T["text"], 700),
           text(36, 76, "The checkpoint is 510 GB on disk and the GPU holds 250 GiB, so part of it stays in CPU memory.", 16, M)]

    def box(x, y, bw, bh, stroke, fill="none", sw=1.5):
        return f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

    out.append(box(36, 100, 540, 250, A, "none", 2))
    out.append(text(56, 128, "GPU MEMORY  /  GB300  /  250 GiB", 13, A, 700, spacing=2))
    out.append(box(56, 144, 300, 186, A, D))
    out.append(text(76, 226, "Hot experts", 24, T["text"], 700))
    out.append(text(76, 254, "291 to 302 of 384 per layer", 16, T["text"]))
    for k, (a, b) in enumerate([("Scratch area", "shared by all layers"), ("Context cache", "0.5M to 1.5M tokens"), ("Everything else", "attention, drafter, head")]):
        y = 144 + k * 64
        out.append(box(372, y, 184, 58, T["accent_bright"] if k == 0 else T["panel_border"], T["bg"]))
        out.append(text(388, y + 25, a, 16, T["text"], 700))
        out.append(text(388, y + 45, b, 12.5, M))
    out.append(box(664, 100, 260, 250, T["panel_border"], "none", 2))
    out.append(text(684, 128, "CPU MEMORY  /  494 GiB", 13, M, 700, spacing=2))
    out.append(box(684, 144, 220, 58, M, T["bg"]))
    out.append(text(700, 169, "Cold experts", 16, T["text"], 700))
    out.append(text(700, 189, "the other 82 to 93 per layer", 12.5, M))
    out.append(box(684, 216, 220, 114, T["panel_border"], T["bg"]))
    out.append(text(700, 266, "Engram tables", 16, T["text"], 700))
    out.append(text(700, 286, "stay in CPU memory", 12.5, M))
    out.append(f'<line x1="684" y1="173" x2="568" y2="173" stroke="{T["accent_bright"]}" stroke-width="2.5"/>')
    out.append(f'<polygon points="556,173 570,165 570,181" fill="{T["accent_bright"]}"/>')
    out.append(text(620, 160, "fetch", 13, T["accent_bright"], 700, "middle"))
    out.append(text(36, 382, "A fetch sits on the critical path of the step. The recipe is about making fetches rare.", 14, M))
    return frame(w, h, out)


def preview(svgs):
    ids = ["stock_vllm", "sglang_patched", "public_recipe", "ours_1m", "ours_262k"]
    rows = ["code_1user", "mixed_1user", "sampled_1user", "users_8", "users_32", "prefill_114k", "ttft_ms"]
    head = "".join(f"<th>{esc(LABEL[i])}</th>" for i in ids)
    body = ""
    for c in rows:
        tds = "".join(f'<td class="{"o" if i.startswith("ours") else ""}">{vtext(CELLS[c][i]) if CELLS[c].get(i) is not None else "not run"}</td>' for i in ids)
        body += f"<tr><td>{esc(CELLS[c]['title'])}</td>{tds}</tr>"
    css = (f"body{{margin:0;background:#0d1117;color:#e6edf3;font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif}}"
           f".w{{max-width:920px;margin:0 auto;padding:28px 24px 60px}}svg{{width:100%;height:auto;display:block;margin:0 0 18px}}"
           f"h2{{font-size:22px;border-bottom:1px solid #30363d;padding-bottom:8px;margin:34px 0 16px}}"
           f"table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #30363d;padding:7px 10px;text-align:right}}"
           f"th:first-child,td:first-child{{text-align:left}}th{{background:#161b22}}td.o{{color:{T['accent_bright']};font-weight:600}}"
           f".b span{{display:inline-block;font-size:12px;margin:0 6px 18px 0;border-radius:3px;overflow:hidden}}"
           f".b i{{font-style:normal;background:#000;color:#fff;padding:4px 8px}}.b b{{font-weight:600;background:{T['accent']};color:#000;padding:4px 8px}}"
           f"p.n{{color:#9198a1;font-size:13px}}")
    badges = "".join(f"<span><i>{a}</i><b>{b}</b></span>" for a, b in
                     [("model", "DeepSeek V4.1 Flash 1017"), ("hardware", "DGX Station, 1x GB300"), ("context", "1M"), ("status", "draft")])
    return (f"<!doctype html><meta charset='utf-8'><title>README preview</title><style>{css}</style><div class='w'>"
            f"{svgs['banner']}<div class='b'>{badges}</div><h2>The numbers</h2>{svgs['one']}{svgs['c32']}"
            f"<table><tr><th></th>{head}</tr>{body}</table>{svgs['scaling']}<h2>How it works</h2>{svgs['memory']}"
            f"<p class='n'>Preview of the README visuals. Colors and banner text: assets/theme.json. Numbers: results/summary.json.</p></div>\n")


def main():
    ids = ["stock_vllm", "sglang_patched", "public_recipe", "ours_1m", "ours_262k"]
    svgs = {
        "banner": banner(),
        "one": hbar("One user, code prompt", "Output tokens per second. Higher is better.", "code_1user", ids,
                    "The faded end of a bar is the spread between boots of the same config."),
        "c32": hbar("32 users at once", "Output tokens per second, all users together. Higher is better.", "users_32", ids),
        "scaling": scaling(),
        "memory": memory_diagram(),
    }
    names = {"banner": "banner.svg", "one": "chart-one-user.svg", "c32": "chart-32-users.svg", "scaling": "chart-scaling.svg",
             "memory": "diagram-memory.svg"}
    for k, n in names.items():
        (ROOT / "assets" / n).write_text(svgs[k])
    (ROOT / "assets" / "preview.html").write_text(preview(svgs))
    print("wrote", ", ".join(names.values()), "and preview.html")


if __name__ == "__main__":
    main()
