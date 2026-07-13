#!/usr/bin/env python3
"""
Run the fragment extraction + canonical-match reconstruction over a folder of game
frames and write a self-contained HTML report:  original | extraction | reconstruction,
plus the matcher's ranked heading candidates and confidences.

The ship is assumed centered in each frame (the game view is ship-centered) at the
canonical pixel scale; an 80x80 window at the frame center is used for matching.

Usage:
  python make_report.py --in-dir heading_hard_cases --out heading_hard_cases/report.html
Needs: numpy, pillow  (use .venv/bin/python).
"""
import argparse, base64, glob, io, json, os, sys
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import match_and_reconstruct as mr   # build_canonical, estimate, render, _shift_rgba, W, _green

W = mr.W

def strict_green(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (g > 110) & (g - r > 60) & (g - b > 50)

def ship_crop(im):
    """Crop an 80x80 window around the detected ship (median of green pixels),
    falling back to the frame center when no ship green is present."""
    a = np.asarray(im).astype(np.int32); m = strict_green(a)
    if m.sum() >= 5:
        ys, xs = np.where(m); cx, cy = int(np.median(xs)), int(np.median(ys))
    else:
        cx, cy = im.width // 2, im.height // 2
    cx = min(max(cx, W // 2), im.width - W // 2)
    cy = min(max(cy, W // 2), im.height - W // 2)
    return im.crop((cx - W // 2, cy - W // 2, cx + W // 2, cy + W // 2)), (cx, cy)

def data_uri(pil_img, scale=1):
    if scale != 1:
        pil_img = pil_img.resize((pil_img.width * scale, pil_img.height * scale), Image.NEAREST)
    buf = io.BytesIO(); pil_img.convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def load_meta(in_dir):
    meta = {}
    p = os.path.join(in_dir, "metadata.jsonl")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line); meta[d.get("frame", "")] = d
            except Exception: pass
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(HERE, "heading_hard_cases"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--pattern", default="*.png")
    args = ap.parse_args()
    out = args.out or os.path.join(args.in_dir, "report.html")

    sprite, cmask = mr.build_canonical()
    sprite_img = Image.fromarray(sprite, "RGBA")
    mask_img = Image.fromarray((cmask * 255).astype(np.uint8), "L")
    meta = load_meta(args.in_dir)

    files = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    files = [f for f in files if not f.endswith("report.html")]
    cards = []
    for f in files:
        full = Image.open(f).convert("RGB")
        crop, (cx, cy) = ship_crop(full); arr = np.asarray(crop).astype(np.int32)
        S = strict_green(arr).astype(np.float32)
        full = full.copy()                       # mark the matched crop region
        ImageDraw.Draw(full).rectangle([cx - W // 2, cy - W // 2, cx + W // 2, cy + W // 2],
                                       outline=(255, 90, 90), width=2)
        name = os.path.basename(f)
        vis = int(S.sum())
        # extraction panel (green on dark)
        ext = np.zeros((W, W, 3), np.uint8); ext[S > 0.5] = (60, 230, 90)
        ext_img = Image.fromarray(ext)
        if vis == 0:
            cards.append((name, full, ext_img, None, None, None, None, meta.get(name)))
            continue
        _, open_mask = mr.open_mask_from_crop(np.asarray(crop))
        res_f = mr.estimate(mask_img, S)                              # fragment + shape (scene-independent)
        res_o = mr.estimate(mask_img, S, open_mask=open_mask, mu=0.2)  # + occluder (gentle shape to avoid conflict)
        def recon(res):
            ship = mr._shift_rgba(sprite_img, res["candidates"][0]["heading"], res["candidates"][0]["shift"])
            rec = np.zeros((W, W, 3), np.uint8); a = ship[..., 3:4] / 255.0
            rec[:] = (ship[..., :3] * a).astype(np.uint8); rec[S > 0.5] = (170, 255, 180)
            return Image.fromarray(rec)
        cards.append((name, full, ext_img, recon(res_f), res_f, recon(res_o), res_o, meta.get(name)))

    # ---- HTML ----
    css = """
    body{background:#12141a;color:#e6e6e6;font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px}
    h1{font-weight:600;font-size:20px} .sub{color:#8a90a0;font-size:13px;margin:4px 0 20px}
    .card{background:#1b1e27;border:1px solid #2a2e3a;border-radius:10px;padding:14px;margin:0 0 16px;display:grid;
          grid-template-columns:380px 150px 185px 185px 1fr;gap:14px;align-items:start}
    .card img{border-radius:6px;display:block;image-rendering:pixelated}
    .lbl{color:#8a90a0;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
    .name{font-weight:600;font-size:13px;margin-bottom:8px;color:#cfd3dd;word-break:break-all}
    .cand{font-size:13px;margin:2px 0} .bar{display:inline-block;height:9px;background:#4a9;border-radius:2px;vertical-align:middle;margin-left:6px}
    .top{color:#ffd34d;font-weight:600} .note{color:#9aa0b0;font-size:12px;margin-top:8px;line-height:1.4}
    .flag{display:inline-block;font-size:11px;padding:2px 7px;border-radius:10px;margin-left:6px}
    .ok{background:#1f4030;color:#7fe0a0} .amb{background:#40361f;color:#e0c060} .low{background:#402020;color:#e08080}
    .cat{display:inline-block;font-size:12px;font-weight:600;padding:4px 10px;border-radius:6px;margin-bottom:8px}
    .cat-self{background:#173d2a;color:#5fe39b;border:1px solid #2f7d54}
    .cat-scene{background:#3d3417;color:#e6c65a;border:1px solid #7d6a2f}
    .cat-under{background:#3d2020;color:#e88;border:1px solid #7d3f3f}
    .catdesc{color:#8a90a0;font-size:11px;margin-bottom:8px}
    .pill{display:inline-block;font-size:12px;padding:3px 9px;border-radius:12px;margin-left:8px}
    @media(max-width:1100px){.card{grid-template-columns:1fr 1fr}}
    """
    def classify(res):
        """Regime from the fragment-only (scene-independent) result."""
        if res is None: return ("under", "no ship", "no ship pixels")
        cs = res["candidates"]; top = cs[0]; second = cs[1] if len(cs) > 1 else None
        if res["visible_px"] < 40:
            return ("under", "underdetermined", "too occluded &mdash; fragment too small")
        if top["conf"] >= 0.6 and not (second and second["conf"] >= 0.3):
            return ("self", "self-sufficient", "fragment shape alone fixes the heading")
        if second and mr._circ(top["heading"], second["heading"]) >= 150 and second["conf"] >= 0.25:
            return ("scene", "needs scene", "axis is clear; bow/stern needs an extra bit (occluder / context)")
        return ("under", "underdetermined", "axis itself ambiguous &mdash; fragment too sparse")
    def cand_html(res):
        if res is None: return '<span class="flag low">no ship pixels</span>'
        cs = res["candidates"]; top = cs[0]
        second = cs[1] if len(cs) > 1 else None
        if res["visible_px"] < 45:      flag = '<span class="flag low">too small</span>'
        elif second and second["conf"] >= 0.4: flag = '<span class="flag amb">axis ok, bow/stern uncertain</span>'
        elif top["conf"] >= 0.6:        flag = '<span class="flag ok">confident</span>'
        else:                            flag = '<span class="flag amb">ambiguous</span>'
        rows = "".join(
            f'<div class="cand"><span class="{ "top" if i==0 else "" }">{c["heading"]:.0f}&deg;</span> '
            f'{c["conf"]:.2f}<span class="bar" style="width:{int(c["conf"]*80)}px"></span></div>'
            for i, c in enumerate(cs) if c["conf"] >= 0.02)
        return f'{flag}<div style="margin-top:6px">{rows}</div>'

    def recon_col(label, rec_img, res):
        if rec_img is None:
            return f'<div><div class="lbl">{label}</div><span class="flag low">no ship</span></div>'
        return (f'<div><div class="lbl">{label}</div>'
                f'<img src="{data_uri(rec_img, 2)}" width="160">{cand_html(res)}</div>')

    body = []
    counts = {"self": 0, "scene": 0, "under": 0}
    for name, full, ext_img, rec_f, res_f, rec_o, res_o, md in cards:
        cat, title, desc = classify(res_f); counts[cat] += 1
        info = (f'<span class="cat cat-{cat}">{title}</span>'
                f'<div class="catdesc">{desc}</div><div class="name">{name}</div>')
        if md:
            parts = []
            if "final_heading_deg" in md: parts.append(f'game final: {md["final_heading_deg"]:.0f}&deg;')
            if res_f: parts.append(f'{res_f["visible_px"]} ship px')
            if md.get("user_note"): parts.append(md["user_note"])
            info += '<div class="note">' + " &middot; ".join(parts) + "</div>"
        body.append(f"""
        <div class="card">
          <div><div class="lbl">original</div><img src="{data_uri(full)}" width="360"></div>
          <div><div class="lbl">extraction</div><img src="{data_uri(ext_img, 2)}" width="150"></div>
          {recon_col("recon &middot; fragment-only", rec_f, res_f)}
          {recon_col("recon &middot; occluder-aware", rec_o, res_o)}
          <div>{info}</div>
        </div>""")

    summary = (f'<span class="pill cat-self">{counts["self"]} self-sufficient</span>'
               f'<span class="pill cat-scene">{counts["scene"]} need scene</span>'
               f'<span class="pill cat-under">{counts["under"]} underdetermined</span>')
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Ship heading — case report</title>
    <style>{css}</style></head><body>
    <h1>Ship heading &mdash; extraction &amp; reconstruction</h1>
    <div class="sub">{len(cards)} cases from {os.path.basename(args.in_dir)}. Heading estimated by matching the
    canonical ship to the extracted fragment (0&deg;=up, clockwise; no orientation prior). Each case is classified
    from the <b>fragment-only</b> result: <b>self-sufficient</b> (the fragment shape alone fixes the heading) vs
    <b>needs scene</b> (axis clear but bow/stern is a coin-flip &mdash; needs the occluder/context bit) vs
    <b>underdetermined</b> (too occluded). The two reconstruction columns are fragment-only (shape) and
    occluder-aware.<br>{summary}</div>
    {''.join(body)}
    </body></html>"""
    open(out, "w").write(html)
    print(f"wrote {out}  ({len(cards)} cases)")

if __name__ == "__main__":
    main()
