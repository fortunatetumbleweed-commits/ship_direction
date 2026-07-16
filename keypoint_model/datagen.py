"""
Synthetic training data for the keypoint detector.

Each sample is a synthetic occluded frame plus, for every keypoint, its projected
(x, y) and a visibility flag (0 if an occluder covers it). Reuses the occluder
compositing from ship_heading_model/datagen.py, and the canonical ship + keypoint
projection from keypoints.py, so labels are generated for free.

Targets for training: per-keypoint Gaussian heatmaps (zero where occluded) and a
per-keypoint visibility vector.

Run `python datagen.py` to write a labelled sample sheet to eyeball.
"""
import os, sys, random
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ship_heading_model"))
import match_and_reconstruct as mr
import datagen as hdg                    # occluder compositing (backgrounds, portrait, text, icons)
import keypoints as kp

W = mr.W
K = len(kp.KP_NAMES)

def assets():
    cmask, sprite, canon = kp.load()
    return sprite, canon, hdg.harvest_backgrounds(), hdg.load_portraits()

def _render_ship(sprite, heading):
    return np.asarray(mr.render(Image.fromarray(sprite, "RGBA"), heading)).astype(np.float32)

def sample(sprite, canon, bgs, ports, occ_prob=0.9, min_vis=2, tries=6):
    """Return (uint8 HxWx3 image, heading, {name:(x,y)}, {name:visible0/1})."""
    heading = random.uniform(0, 360)
    jx, jy = random.randint(-4, 4), random.randint(-4, 4)
    ship = np.roll(np.roll(_render_ship(sprite, heading), jy, 0), jx, 1)
    ship_a = ship[..., 3:4] / 255.0
    kps = kp.project(canon, heading, shift=(jy, jx))
    best = None
    for _ in range(tries):
        img = hdg._bg_canvas(bgs)
        img[:] = ship[..., :3] * ship_a + img * (1 - ship_a)
        occ = np.zeros((W, W), np.float32)
        if random.random() < occ_prob:
            for rgb, m in hdg._occluder_layers(bgs, ports):
                m3 = m[..., None]; img[:] = rgb * m3 + img * (1 - m3); occ = np.maximum(occ, m)
        vis = {}
        for name, (x, y) in kps.items():
            xi, yi = int(round(x)), int(round(y))
            vis[name] = 1.0 if (0 <= xi < W and 0 <= yi < W and occ[yi, xi] < 0.5) else 0.0
        nvis = sum(vis.values())
        if best is None or nvis > best[0]:
            best = (nvis, img.copy(), vis)
        if nvis >= min_vis:
            break
    _, img, vis = best
    img += np.random.randn(W, W, 3) * random.uniform(0, 4)
    return img.clip(0, 255).astype(np.uint8), heading, kps, vis

def heatmaps(kps, vis, sigma=2.0):
    """(K,W,W) Gaussian heatmaps (zero where occluded) and (K,) visibility vector."""
    H = np.zeros((K, W, W), np.float32)
    yy, xx = np.mgrid[0:W, 0:W]
    for i, name in enumerate(kp.KP_NAMES):
        if vis[name] > 0.5:
            x, y = kps[name]
            H[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    v = np.array([vis[n] for n in kp.KP_NAMES], np.float32)
    return H, v

COLORS = {"bow": (255, 60, 60), "stern_l": (80, 160, 255), "stern_r": (80, 220, 255),
          "fin_up_l": (255, 230, 40), "fin_up_r": (255, 170, 40),
          "fin_lo_l": (150, 255, 80), "fin_lo_r": (60, 210, 120), "center": (230, 230, 230)}

if __name__ == "__main__":
    from PIL import ImageDraw, ImageFont
    random.seed(0); np.random.seed(0)
    sprite, canon, bgs, ports = assets()
    print(f"keypoints: {K}  bg tiles: {len(bgs)}  portraits: {len(ports)}")
    n, cols, Z, pad = 24, 6, 3, 4; rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (W * Z + pad) + pad, rows * (W * Z + pad) + pad), (20, 20, 20))
    for i in range(n):
        img, heading, kps, vis = sample(sprite, canon, bgs, ports)
        im = Image.fromarray(img).resize((W * Z, W * Z), Image.NEAREST); d = ImageDraw.Draw(im)
        for name, (x, y) in kps.items():
            if vis[name] > 0.5:                       # solid dot = visible (a training target)
                d.ellipse([x * Z - 3, y * Z - 3, x * Z + 3, y * Z + 3], fill=COLORS[name])
        sheet.paste(im, (pad + (i % cols) * (W * Z + pad), pad + (i // cols) * (W * Z + pad)))
    sheet.save(os.path.join(HERE, "datagen_preview.png"))
    print("wrote datagen_preview.png  (solid dots = visible keypoints = training targets)")
