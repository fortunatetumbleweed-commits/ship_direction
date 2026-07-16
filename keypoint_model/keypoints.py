"""
Keypoint definitions for the ship, in the canonical frame (heading 0 = bow up).

The ship has distinctive parts, each of which independently indicates direction:
  bow       - the pointy nose (points in the heading direction)
  stern_l   - left corner of the flat transom (back)
  stern_r   - right corner of the flat transom
  fin_up_l  - upper sail/fin tip, left  (forward pair)
  fin_up_r  - upper sail/fin tip, right
  fin_lo_l  - lower sail/fin tip, left   (widest pair)
  fin_lo_r  - lower sail/fin tip, right
  center    - hull centroid (position anchor; not directional on its own)

Keypoints are extracted geometrically from the canonical ship mask, then can be
projected to any (heading, shift) pose to make training targets or to solve pose
from detected points.

Run `python keypoints.py` to (re)build the canonical assets and write a preview.
"""
import os, sys, json, math
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # repo root, for match_and_reconstruct
import match_and_reconstruct as mr                  # build_canonical, render, W, CENTER

W = mr.W
KP_NAMES = ["bow", "stern_l", "stern_r", "fin_up_l", "fin_up_r", "fin_lo_l", "fin_lo_r", "center"]

def canonical_keypoints(mask):
    """Extract keypoint (x, y) coords from the canonical ship mask (heading 0, bow up)."""
    ys, xs = np.where(mask)
    top = ys.min(); bot = ys.max()
    def band(rows):                                  # mean x of pixels in a set of rows
        sel = np.isin(ys, rows)
        return float(xs[sel].mean()), float(ys[sel].mean())
    bx, by = band(range(top, top + 3))               # bow: mean of the top 3 rows
    # stern transom: the bottom band; its left/right extremes are the corners
    stern_rows = [r for r in range(bot - 2, bot + 1)]
    sm = np.isin(ys, stern_rows)
    sxs, sys_ = xs[sm], ys[sm]
    stern_l = (float(sxs.min()), float(sys_[sxs.argmin()]))
    stern_r = (float(sxs.max()), float(sys_[sxs.argmax()]))
    # fin/sail tips: two lateral protrusions (upper + lower). Find the two rows of
    # peak half-width, and place the left/right tip at each row's horizontal extremes.
    rows = [r for r in range(top, bot + 1) if (ys == r).any()]
    minx = {r: int(xs[ys == r].min()) for r in rows}
    maxx = {r: int(xs[ys == r].max()) for r in rows}
    ext = {r: (maxx[r] - minx[r]) / 2 for r in rows}
    peaks = [r for r in rows if ext[r] >= ext.get(r - 1, -1) and ext[r] >= ext.get(r + 1, -1)]
    peaks.sort(key=lambda r: -ext[r]); sel = []
    for r in peaks:
        if all(abs(r - s) >= 5 for s in sel): sel.append(r)
        if len(sel) == 2: break
    up, lo = sorted(sel)                             # upper (smaller y) then lower
    center = (float(xs.mean()), float(ys.mean()))
    return {"bow": (bx, by), "stern_l": stern_l, "stern_r": stern_r,
            "fin_up_l": (float(minx[up]), float(up)), "fin_up_r": (float(maxx[up]), float(up)),
            "fin_lo_l": (float(minx[lo]), float(lo)), "fin_lo_r": (float(maxx[lo]), float(lo)),
            "center": center}

def project(kps, heading, shift=(0, 0)):
    """Rotate canonical keypoints to `heading` (0=up, cw) about CENTER, then translate."""
    cx, cy = mr.CENTER; a = math.radians(heading); ca, sa = math.cos(a), math.sin(a)
    out = {}
    for name, (x, y) in kps.items():
        dx, dy = x - cx, y - cy
        # clockwise rotation on screen (y-down) by `heading`, matching mr.render(-heading)
        rx = ca * dx - sa * dy
        ry = sa * dx + ca * dy
        out[name] = (cx + rx + shift[1], cy + ry + shift[0])
    return out

def load():
    """Return (canonical_mask bool, canonical colored sprite uint8, keypoints dict)."""
    sprite, cmask = mr.build_canonical()
    return cmask, sprite, canonical_keypoints(cmask)

if __name__ == "__main__":
    from PIL import ImageDraw, ImageFont
    cmask, sprite, kps = load()
    json.dump({k: list(v) for k, v in kps.items()}, open(os.path.join(HERE, "keypoints.json"), "w"), indent=2)
    print("canonical keypoints (x, y):")
    for k in KP_NAMES: print(f"  {k:8s} ({kps[k][0]:5.1f}, {kps[k][1]:5.1f})")

    COLORS = {"bow": (255, 60, 60), "stern_l": (80, 160, 255), "stern_r": (80, 220, 255),
              "fin_up_l": (255, 230, 40), "fin_up_r": (255, 170, 40),
              "fin_lo_l": (150, 255, 80), "fin_lo_r": (60, 210, 120), "center": (210, 210, 210)}
    def fnt(s=11):
        try: return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", s)
        except Exception: return ImageFont.load_default()
    def draw_pose(heading):
        base = np.zeros((W, W, 3), np.uint8)
        sh = np.asarray(mr.render(Image.fromarray(sprite, "RGBA"), heading))
        a = sh[..., 3:4] / 255.0; base[:] = (sh[..., :3] * a).astype(np.uint8)
        im = Image.fromarray(base); d = ImageDraw.Draw(im)
        for name, (x, y) in project(kps, heading).items():
            d.ellipse([x - 1.6, y - 1.6, x + 1.6, y + 1.6], fill=COLORS[name])
        return im
    Z, pad = 6, 6; poses = [0, 45, 120, 210, 300]
    sheet = Image.new("RGB", (len(poses) * (W * Z + pad) + pad, W * Z + pad + 16), (20, 20, 20))
    dd = ImageDraw.Draw(sheet)
    for i, h in enumerate(poses):
        sheet.paste(draw_pose(h).resize((W * Z, W * Z), Image.NEAREST), (pad + i * (W * Z + pad), 16))
        dd.text((pad + i * (W * Z + pad) + 2, 2), f"heading {h}", font=fnt(12), fill=(230, 230, 230))
    # legend
    lx = pad
    for name in KP_NAMES:
        dd.text((lx, W * Z + 2), name, font=fnt(11), fill=COLORS[name]); lx += 70
    sheet.save(os.path.join(HERE, "keypoints_preview.png"))
    print("wrote keypoints.json, keypoints_preview.png")
