#!/usr/bin/env python3
"""
Match a canonical ship to an occluded ship FRAGMENT (no orientation prior) and, from
the match, reconstruct the whole ship in the same orientation. Shows both paths:

  Path 1  MATCH -> direction : rotate the canonical ship through all 360 deg, register
                              it onto the fragment, keep the best-scoring heading.
  Path 2  RECONSTRUCT        : render the full canonical ship at that heading/position,
                              overlay the fragment to show the de-occluded whole ship.

The ship is left/right symmetric but front/back asymmetric, so the AXIS is reliable
while bow-vs-stern can be ambiguous for tiny fragments; the tool reports a confidence
and flags 'direction uncertain' when heading vs heading+180 score alike.

Input: fragment PNGs (RGBA cutouts from extract_ship_fragments.py; alpha = ship). The
ship icon is assumed a fixed scale on an 80x80 canvas (as produced by the extractor).
Any truthNNN in a filename is shown for reference only -- it is NEVER used to estimate.

Usage:
  python match_and_reconstruct.py ship_fragments/*_fragment.png --out-dir match_out
Needs: numpy, pillow (use ../.venv/bin/python).
"""
import argparse, glob, os, re, math, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "heading_v9d_raw_images")
W = 80; CENTER = (40.0, 40.0)

# ---------- canonical ship (heading 0) ----------
def _rgb(f): return np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
def _green(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (g > 90) & (g > r + 30) & (g > b + 30)

def build_canonical(raw_dir=RAW_DIR):
    """Return (colored_sprite_uint8 RGBA, mask_bool) at heading 0, L/R folded."""
    rgbs, alphas = [], []
    for f in sorted(glob.glob(os.path.join(raw_dir, "synth_*truth*.png"))):
        t = int(re.search(r"truth(\d+)", f).group(1))
        a = _rgb(f); m = _green(a).astype(np.float32)
        der = Image.fromarray(np.dstack([a, m * 255]).astype(np.uint8), "RGBA").rotate(
            +t, resample=Image.BICUBIC, center=CENTER)
        d = np.asarray(der).astype(np.float32); rgbs.append(d[..., :3]); alphas.append(d[..., 3])
    A = np.stack(alphas); mA = np.median(A, 0)
    mC = (np.stack(rgbs) * (A > 60)[..., None]).sum(0) / np.maximum((A > 60).sum(0), 1)[..., None]
    sprite = np.maximum(np.dstack([mC, mA]), np.fliplr(np.dstack([mC, mA])))
    sprite = sprite.clip(0, 255).astype(np.uint8)
    return sprite, sprite[..., 3] > 60

def render(img_or_arr, angle):
    """Rotate an RGBA sprite (or L mask image) to `angle` (0=up, clockwise)."""
    return img_or_arr.rotate(-angle % 360, resample=Image.BICUBIC, center=CENTER)

# ---------- matching (no orientation prior) ----------
def _xcorr_peak(S, T):
    """Best circular shift of T onto S; return (overlap, dy, dx)."""
    c = np.fft.irfft2(np.fft.rfft2(S) * np.conj(np.fft.rfft2(T)), s=S.shape)
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    dy = iy - (S.shape[0] if iy > S.shape[0] // 2 else 0)
    dx = ix - (S.shape[1] if ix > S.shape[1] // 2 else 0)
    return float(c[iy, ix]), dy, dx

def _circ(a, b, m=360): return abs((a - b + m / 2) % m - m / 2)

def _depth(mask_bool):
    """Thickness map: each pixel's value = erosions until it vanishes (rotation-invariant)."""
    d = np.zeros(mask_bool.shape, np.float32); cur = (mask_bool.astype(np.uint8) * 255)
    for k in range(1, 25):
        er = np.asarray(Image.fromarray(cur).filter(ImageFilter.MinFilter(3)))
        d[(cur > 127) & (er <= 127)] = k; cur = er
        if (cur > 127).sum() == 0: break
    return d

def open_mask_from_crop(crop_rgb):
    """Split a crop into occluder vs open background. Returns (occluder_bool, open_bool).

    Occluders are the bright, non-terrain overlays the ship sails under (portrait, text,
    icons); they're solidified by morphological close + hole-fill so a hull pixel may
    legitimately hide beneath them. 'open' = water/terrain where a hull pixel would be
    visible if present (and so must NOT be covered by the reconstructed ship).
    """
    a = crop_rgb.astype(np.int32); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    ship = (g > 110) & (g - r > 60) & (g - b > 50)
    yellow = (r > 195) & (g > 160) & (r - b > 90) & (g - b > 60)   # saturated portrait yellow (not tan terrain)
    white = (r > 200) & (g > 200) & (b > 200)                       # text / diamonds / house markers
    ui = Image.fromarray(((yellow | white) * 255).astype(np.uint8))
    ui = ui.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))     # close blobs
    inv = Image.eval(ui, lambda p: 255 - p)                                       # fill interior holes
    for c in [(0, 0), (ui.width - 1, 0), (0, ui.height - 1), (ui.width - 1, ui.height - 1)]:
        ImageDraw.floodfill(inv, c, 0)
    occ = (np.asarray(ui) > 127) | (np.asarray(inv) > 127)
    occ = np.asarray(Image.fromarray((occ * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 127
    return occ, (~occ) & (~ship)

def estimate(mask_img, S, open_mask=None, lam=3.0, mu=0.5, max_candidates=4, sep=35, temp=0.03):
    """Match canonical mask to fragment mask S over all 360 deg (no orientation prior).

    Returns ranked candidate headings, each with a pose and a confidence (softmax over
    peak match scores). A clear fragment -> one dominant candidate; a bow/stern-ambiguous
    one -> theta and theta+180 near 50/50; a tiny fragment -> probability spread out.

    Two shape/scene refinements resolve ties the raw overlap cannot:
    - `mu`  (always on, scene-independent): penalize the fragment sitting DEEPER inside
       the template than its own thickness -- i.e. a pointy tip crammed into a blob. It
       must line up with a matching-width part of the hull (bow tip -> bow tip).
    - `open_mask` (scene-aware, optional): True where open water/terrain; poses whose
       hidden hull would fall on open background are penalized (it can only hide under
       an occluder). Resolves solid-occluder tip cases.
    """
    nS = float(S.sum()); nT = float((np.asarray(mask_img) > 127).sum())
    FS = np.fft.rfft2(S)
    FO = np.fft.rfft2(open_mask.astype(np.float32)) if open_mask is not None else None
    Sb = S > 0.5
    fdep = _depth(Sb)                                                  # fragment's own thickness
    depth0 = Image.fromarray((_depth(np.asarray(mask_img) > 127) * 10).clip(0, 255).astype(np.uint8), "L")
    def score_at(a):
        # jointly pick the shift maximizing coverage x fill x occluder-consistency ...
        T = (np.asarray(render(mask_img, a)).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        FT = np.conj(np.fft.rfft2(T))
        inter = np.maximum(np.fft.irfft2(FS * FT, s=S.shape), 0.0)     # fragment overlap, all shifts
        sc = (inter / max(nS, 1)) * (inter / max(nT, 1)) ** 0.3
        if FO is not None:
            viol = np.maximum(np.fft.irfft2(FO * FT, s=S.shape), 0.0)  # hull over open water, all shifts
            sc = sc * np.exp(-lam * viol / max(nT, 1))
        iy, ix = np.unravel_index(np.argmax(sc), sc.shape)
        dy = iy - (S.shape[0] if iy > S.shape[0] // 2 else 0)
        dx = ix - (S.shape[1] if ix > S.shape[1] // 2 else 0)
        s = float(sc[iy, ix])
        if mu > 0 and Sb.any():                                       # ... then the shape term
            td = np.roll(np.roll(np.asarray(render(depth0, a)).astype(np.float32) / 10.0, dy, 0), dx, 1)
            s *= math.exp(-mu * float(np.maximum(td[Sb] - fdep[Sb], 0.0).mean()))
        return s, dy, dx
    prof = {a: score_at(a) for a in range(0, 360, 3)}            # score profile over headings
    picks = []                                                   # non-max suppression of peaks
    for a in sorted(prof, key=lambda k: -prof[k][0]):
        if all(_circ(a, pa) >= sep for pa in picks):
            picks.append(a)
        if len(picks) >= max_candidates: break
    cands = []
    for a in picks:                                             # refine each peak to 1 deg
        best = (prof[a][0], a, prof[a][1], prof[a][2])
        for aa in range(a - 3, a + 4):
            s, dy, dx = score_at(aa % 360)
            if s > best[0]: best = (s, aa % 360, dy, dx)
        cands.append(best)
    scr = np.array([c[0] for c in cands])
    p = np.exp((scr - scr.max()) / temp); p = p / p.sum()
    order = np.argsort(-p)
    candidates = [{"heading": cands[i][1], "axis": cands[i][1] % 180,
                   "shift": (cands[i][2], cands[i][3]),
                   "score": round(float(cands[i][0]), 3), "conf": round(float(p[i]), 2)}
                  for i in order]
    return {"visible_px": int(nS), "candidates": candidates}

# ---------- rendering the results ----------
def _shift_rgba(sprite_img, angle, shift):
    arr = np.asarray(render(sprite_img, angle)).astype(np.uint8)
    dy, dx = shift
    return np.roll(np.roll(arr, dy, 0), dx, 1)

def _font(sz=11):
    try: return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", sz)
    except Exception: return ImageFont.load_default()

def _arrow(draw, cx, cy, heading, length, fill, width=2):
    d = (math.sin(math.radians(heading)), -math.cos(math.radians(heading)))
    hx, hy = cx + d[0] * length, cy + d[1] * length
    draw.line([(cx, cy), (hx, hy)], fill=fill, width=width)
    left = (math.radians(heading + 150)); right = (math.radians(heading - 150))
    for ang in (left, right):
        draw.line([(hx, hy), (hx + math.sin(ang) * 6, hy - math.cos(ang) * 6)], fill=fill, width=width)

def panels(sprite, mask_img, S, res):
    """Return three 80x80 RGB panels: [fragment | direction (all candidates) | reconstruction (top)]."""
    from PIL import ImageFilter
    cands = res["candidates"]; top = cands[0]
    green = np.zeros((W, W, 3), np.uint8); green[S > 0.5] = (60, 230, 90)
    p_frag = Image.fromarray(green)

    # direction panel: top candidate's outline + one arrow per candidate (brightness/width by conf)
    ship = _shift_rgba(sprite, top["heading"], top["shift"])
    alpha = ship[..., 3] > 60
    er = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))) > 127
    dir_im = np.zeros((W, W, 3), np.uint8); dir_im[alpha & ~er] = (0, 120, 140); dir_im[S > 0.5] = (60, 230, 90)
    p_dir = Image.fromarray(dir_im); dd = ImageDraw.Draw(p_dir)
    for c in reversed(cands):                        # draw weakest first so top is on top
        conf = c["conf"]; b = int(80 + 175 * conf)
        cx, cy = 40 + c["shift"][1], 40 + c["shift"][0]
        _arrow(dd, cx, cy, c["heading"], 16 + 8 * conf, (b, int(b * 0.8), 0), width=1 + int(round(2 * conf)))
    for i, c in enumerate(cands[:3]):
        dd.text((2, 2 + i * 11), f"{c['heading']:.0f} {c['conf']:.2f}", font=_font(11),
                fill=(255, 255, 255) if i == 0 else (185, 185, 185))

    recon = np.zeros((W, W, 3), np.uint8); a = ship[..., 3:4] / 255.0
    recon[:] = (ship[..., :3] * a).astype(np.uint8)
    recon[S > 0.5] = (170, 255, 180)                 # real fragment highlighted on the full ship
    p_rec = Image.fromarray(recon)
    return p_frag, p_dir, p_rec

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fragments", nargs="+", help="fragment PNGs (RGBA cutouts; alpha=ship)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "match_out"))
    ap.add_argument("--scale", type=int, default=4, help="montage upscale")
    args = ap.parse_args()
    sprite, cmask = build_canonical()
    sprite_img = Image.fromarray(sprite, "RGBA")
    mask_img = Image.fromarray((cmask * 255).astype(np.uint8), "L")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    print(f"{'fragment':30s} {'vis':>4}  candidates: heading(conf) ...")
    for f in args.fragments:
        im = Image.open(f).convert("RGBA"); arr = np.asarray(im)
        if arr.shape[:2] != (W, W):                # center onto an 80x80 canvas (fixed scale)
            canvas = Image.new("RGBA", (W, W), (0, 0, 0, 0))
            canvas.paste(im, ((W - im.width) // 2, (W - im.height) // 2)); arr = np.asarray(canvas)
        S = (arr[..., 3] > 128).astype(np.float32) if arr.shape[2] == 4 and arr[..., 3].max() > 0 \
            else _green(arr[..., :3].astype(np.float32)).astype(np.float32)
        res = estimate(mask_img, S)
        name = os.path.splitext(os.path.basename(f))[0]
        pf, pd, pr = panels(sprite_img, mask_img, S, res)
        for k, c in enumerate(res["candidates"]):
            if c["conf"] >= 0.2:                    # save a reconstruction per real candidate
                r = _shift_rgba(sprite_img, c["heading"], c["shift"])
                Image.fromarray(r, "RGBA").save(
                    os.path.join(args.out_dir, f"{name}_cand{k+1}_h{c['heading']:.0f}_p{c['conf']:.2f}.png"))
        cand_str = "  ".join(f"{c['heading']:.0f}({c['conf']:.2f})" for c in res["candidates"])
        m = re.search(r"truth(\d+)", name)
        ref = f"   [truth {int(m.group(1))}, not used]" if m else ""
        print(f"{name:30s} {res['visible_px']:4d}  {cand_str}{ref}")
        rows.append((name, pf, pd, pr))

    # montage: [fragment | direction | reconstruction]
    Z = args.scale; pad = 6; head = 16
    sheet = Image.new("RGB", (3 * W * Z + 4 * pad, len(rows) * (W * Z + pad) + pad + head), (28, 28, 28))
    dh = ImageDraw.Draw(sheet); fnt = _font(12)
    for i, (lbl, txt) in enumerate([("fragment", 0), ("Path1: direction", 1), ("Path2: reconstruction", 2)]):
        dh.text((pad + txt * (W * Z + pad), 3), lbl, font=fnt, fill=(220, 220, 220))
    for i, (name, pf, pd, pr) in enumerate(rows):
        y = head + pad + i * (W * Z + pad)
        for j, p in enumerate((pf, pd, pr)):
            sheet.paste(p.resize((W * Z, W * Z), Image.NEAREST), (pad + j * (W * Z + pad), y))
    sheet.save(os.path.join(args.out_dir, "_montage.png"))
    print(f"\nsaved reconstructions + _montage.png to {args.out_dir}")

if __name__ == "__main__":
    main()
