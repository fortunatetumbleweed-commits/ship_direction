"""
Generate the README figures. Uses the standalone library itself, so the docs always
reflect the shipped code.

Run:  python ship_parts/make_docs.py   ->  ship_parts/docs/*.png
"""
import os, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ship_parts import ShipHeading, PART_NAMES
from ship_parts.estimator import render_labels, render_sprite

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
RAW = os.path.join(os.path.dirname(HERE), "heading_v9d_raw_images")
BG = (18, 18, 20)
PCOL = {1: (255, 70, 70), 2: (155, 155, 160), 3: (70, 150, 255), 4: (255, 220, 50), 5: (255, 140, 40)}

def fnt(s=12, bold=True):
    p = f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf"
    try: return ImageFont.truetype(p, s)
    except Exception: return ImageFont.load_default()

def colorize(lab, base=None):
    im = (base // 3 if base is not None else np.zeros((80, 80, 3), np.uint8)).copy()
    for p in range(1, 6): im[lab == p] = PCOL[p]
    return im

def up(a, z=4): return Image.fromarray(a).resize((80 * z, 80 * z), Image.NEAREST)

# ---------------------------------------------------------------- 1. the 5 parts
def fig_parts(est):
    z, pad, hdr = 4, 10, 26
    tiles = []
    ship0 = render_sprite(est.sprite, 0)
    a = ship0[..., 3:4] / 255.0
    tiles.append(("ship (heading 0)", (ship0[..., :3] * a).astype(np.uint8)))
    for h in (0, 60, 150, 270):
        tiles.append((f"parts @ {h}", colorize(render_labels(est.L0, h))))
    Wd = len(tiles) * (80 * z + pad) + pad
    im = Image.new("RGB", (Wd, hdr + 80 * z + 46), BG); d = ImageDraw.Draw(im)
    d.text((pad, 6), "The ship is carved into 5 whole regions — each rotates rigidly with the hull",
           fill=(240, 240, 240), font=fnt(14))
    for i, (lab, arr) in enumerate(tiles):
        x = pad + i * (80 * z + pad)
        im.paste(up(arr, z), (x, hdr))
        d.text((x + 2, hdr + 80 * z + 4), lab, fill=(215, 215, 215), font=fnt(11, False))
    lx = pad
    for p in range(1, 6):
        d.rectangle([lx, hdr + 80 * z + 24, lx + 12, hdr + 80 * z + 36], fill=PCOL[p], outline=(230, 230, 230))
        d.text((lx + 17, hdr + 80 * z + 24), PART_NAMES[p], fill=(225, 225, 225), font=fnt(12))
        lx += 17 + 8 * len(PART_NAMES[p]) + 18
    im.save(os.path.join(DOCS, "01-parts.png")); print("wrote 01-parts.png")

# ---------------------------------------------------------------- 2. pipeline
def fig_pipeline(est):
    cases = [("synth_s02_truth205.png", "clean"),
             ("hard_t557_truth347.png", "light occlusion"),
             ("hard_t082_truth168.png", "only the stern visible"),
             ("hard_t083_truth174.png", "only the bow visible")]
    z, pad, hdr = 4, 10, 30
    cols = ["input frame", "U-Net: predicted parts", "fit + reconstruct"]
    tile = 80 * z
    im = Image.new("RGB", (3 * (tile + pad) + pad, hdr + len(cases) * (tile + pad + 18) + pad), BG)
    d = ImageDraw.Draw(im)
    for c, t in enumerate(cols):
        d.text((pad + c * (tile + pad), 8), t, fill=(240, 240, 240), font=fnt(14))
    for r, (fn, note) in enumerate(cases):
        img = np.asarray(Image.open(os.path.join(RAW, fn)).convert("RGB"))
        truth = int(re.search(r"truth(\d+)", fn).group(1))
        res = est(img)
        y = hdr + r * (tile + pad + 18)
        for c, arr in enumerate((img, colorize(res.labels, img), est.reconstruct(img, res))):
            im.paste(up(arr, z), (pad + c * (tile + pad), y))
        err = abs((res.heading - truth + 180) % 360 - 180)
        d.text((pad, y + tile + 3), f"{fn.split('_')[1]} — {note}", fill=(215, 215, 215), font=fnt(11, False))
        d.text((pad + (tile + pad), y + tile + 3), "sees: " + (", ".join(res.parts_seen) or "-"),
               fill=(215, 215, 215), font=fnt(11, False))
        d.text((pad + 2 * (tile + pad), y + tile + 3),
               f"heading {res.heading:.0f}   truth {truth}   err {err:.0f}",
               fill=(140, 230, 140) if err <= 20 else (240, 140, 140), font=fnt(11))
    im.save(os.path.join(DOCS, "02-pipeline.png")); print("wrote 02-pipeline.png")

if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    est = ShipHeading()
    fig_parts(est); fig_pipeline(est)
