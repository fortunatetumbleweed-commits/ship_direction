"""
Diagnostic visualizations for the keypoint detector. Regenerates:
  feature_slices.png    - exactly how each feature is DEFINED: the Gaussian training
                          target for each keypoint, drawn on the canonical ship (shows
                          how small/point-like each "feature" is vs the actual part).
  kp_features.png       - the model's detected-feature heatmaps on clean/occluded images.
  kp_detect.png         - constellation-fit result on the real hard images (reconstruction
                          + detected part dots + heading error).
  kp_training_samples.png - synthetic training frames, most-occluded first.

Run:  python viz.py
"""
import os, glob, re, random, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import keypoints as kp
import match_and_reconstruct as mr
import datagen as kdg
from train import KeypointNet, to_tensor, heatmaps_conf, detect, DEVICE, W

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "heading_v9d_raw_images")
SIGMA = 2.0                                     # must match datagen heatmap sigma
COL = kdg.COLORS
def fnt(s=11):
    try: return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", s)
    except Exception: return ImageFont.load_default()
def circ(a, b): return abs((a - b + 180) % 360 - 180)

def _model():
    ck = torch.load(os.path.join(HERE, "kp_model_v1.pt"), map_location=DEVICE)
    m = KeypointNet().to(DEVICE); m.load_state_dict(ck["state_dict"]); m.eval(); return m

def _gauss(cx, cy, sigma=SIGMA):
    yy, xx = np.mgrid[0:W, 0:W]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

# ---- 1. how each feature is defined (the "slice") ----
def feature_slices():
    cmask, sprite, canon = kp.load()
    ship = (np.dstack([np.zeros((W, W, 3)), cmask * 255])).astype(np.uint8)
    Z, pad, cols = 6, 6, 4; names = kp.KP_NAMES; rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (W * Z + pad) + pad, rows * (W * Z + pad) + pad + 16), (16, 16, 16))
    dd = ImageDraw.Draw(sheet)
    dd.text((pad, 2), f"how each feature is DEFINED (Gaussian target, sigma={SIGMA}px) on the canonical ship",
            font=fnt(12), fill=(230, 230, 230))
    for i, nm in enumerate(names):
        x, y = canon[nm]; g = _gauss(x, y)
        base = (cmask[..., None] * np.array([40, 90, 40])).astype(np.float32)      # dim ship
        base[..., 0] += g * COL[nm][0]; base[..., 1] += g * COL[nm][1]; base[..., 2] += g * COL[nm][2]
        im = Image.fromarray(base.clip(0, 255).astype(np.uint8)).resize((W * Z, W * Z), Image.NEAREST)
        d = ImageDraw.Draw(im)
        npx = int((g > 0.5).sum())                                                # half-max footprint
        d.text((2, 2), nm, font=fnt(12), fill=COL[nm])
        d.text((2, W * Z - 14), f"{npx}px > 0.5", font=fnt(11), fill=(210, 210, 210))
        sheet.paste(im, (pad + (i % cols) * (W * Z + pad), 16 + pad + (i // cols) * (W * Z + pad)))
    sheet.save(os.path.join(HERE, "feature_slices.png")); print("wrote feature_slices.png")

# ---- 2. detected heatmaps on a few images ----
def features_montage(m, canon):
    picks = [("synth_s02_truth205", "clean"), ("hard_t557_truth347", "light occ"),
             ("hard_t104_truth207", "occ"), ("hard_t082_truth168", "heavy occ"),
             ("hard_t083_truth210", "heavy occ")]
    Z, pad = 5, 6
    sheet = Image.new("RGB", (2 * W * Z + 3 * pad, len(picks) * (W * Z + pad) + pad), (16, 16, 16))
    DD = ImageDraw.Draw(sheet)
    for i, (name, tag) in enumerate(picks):
        f = glob.glob(os.path.join(RAW, name + ".png"))[0]; truth = int(re.search(r"truth(\d+)", name).group(1))
        img = np.asarray(Image.open(f).convert("RGB")); hm, cf = heatmaps_conf(m, to_tensor(img[None]))
        heading, _ = kp.constellation_pose(hm[0], cf[0], canon)
        over = img.astype(np.float32) * 0.35
        for k, nm in enumerate(kp.KP_NAMES):
            h = hm[0, k]; h = h / max(h.max(), 1e-6)
            for c in range(3): over[..., c] += h * COL[nm][c] * min(cf[0, k] / 0.5, 1.0)
        y = pad + i * (W * Z + pad)
        sheet.paste(Image.fromarray(img).resize((W * Z, W * Z), Image.NEAREST), (pad, y))
        sheet.paste(Image.fromarray(over.clip(0, 255).astype(np.uint8)).resize((W * Z, W * Z), Image.NEAREST), (2 * pad + W * Z, y))
        DD.text((pad + 2, y + 2), f"{name.split('_')[1] if name.startswith('hard') else name.split('_')[0]} {tag}", font=fnt(11), fill=(240, 240, 240))
        DD.text((2 * pad + W * Z + 2, y + 2), f"truth {truth}  pred {heading:.0f}  err {circ(heading, truth):.0f}", font=fnt(11), fill=(240, 240, 240))
    sheet.save(os.path.join(HERE, "kp_features.png")); print("wrote kp_features.png")

# ---- 3. constellation reconstruction on all hard images ----
def detect_montage(m, canon):
    sprite, _ = mr.build_canonical()
    files = sorted(glob.glob(os.path.join(RAW, "hard_*truth*.png"))); Z, pad = 5, 6
    sheet = Image.new("RGB", (len(files) * (W * Z + pad) + pad, W * Z + pad + 16), (16, 16, 16)); DD = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        truth = int(re.search(r"truth(\d+)", f).group(1)); img = np.asarray(Image.open(f).convert("RGB"))
        hm, cf = heatmaps_conf(m, to_tensor(img[None])); heading, shift = kp.constellation_pose(hm[0], cf[0], canon)
        ship = np.roll(np.roll(np.asarray(mr.render(Image.fromarray(sprite, "RGBA"), heading)), shift[0], 0), shift[1], 1)
        a = ship[..., 3:4] / 255.0; rec = (img * (1 - a) + ship[..., :3] * a).astype(np.uint8)
        im = Image.fromarray(rec).resize((W * Z, W * Z), Image.NEAREST); d = ImageDraw.Draw(im)
        det, conf = detect(m, to_tensor(img[None]))[0]
        for nm, (x, y) in det.items():
            if conf[nm] >= 0.3: d.ellipse([x * Z - 2, y * Z - 2, x * Z + 2, y * Z + 2], fill=COL[nm], outline=(0, 0, 0))
        sheet.paste(im, (pad + i * (W * Z + pad), 16))
        DD.text((pad + i * (W * Z + pad) + 2, 2), f"{os.path.basename(f).split('_')[1]} t{truth} p{heading:.0f} e{circ(heading, truth):.0f}", font=fnt(10), fill=(230, 230, 230))
    sheet.save(os.path.join(HERE, "kp_detect.png")); print("wrote kp_detect.png")

# ---- 4. training samples (most-occluded first) ----
def training_samples():
    random.seed(7); np.random.seed(7)
    sprite, canon, bgs, ports = kdg.assets(); samps = []
    for _ in range(60):
        img, h, kps, vis = kdg.sample(sprite, canon, bgs, ports); samps.append((int(sum(vis.values())), img, kps, vis))
    samps.sort(key=lambda s: s[0]); pick = samps[:18] + samps[-6:]
    Z, pad, cols = 4, 5, 6; rows = (len(pick) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (W * Z + pad) + pad, rows * (W * Z + pad) + pad), (16, 16, 16))
    for i, (nv, img, kps, vis) in enumerate(pick):
        im = Image.fromarray(img).resize((W * Z, W * Z), Image.NEAREST); d = ImageDraw.Draw(im)
        for nm, (x, y) in kps.items():
            X, Y = x * Z, y * Z
            if vis[nm] > 0.5: d.ellipse([X - 3.5, Y - 3.5, X + 3.5, Y + 3.5], fill=COL[nm], outline=(0, 0, 0))
            else: d.ellipse([X - 2.5, Y - 2.5, X + 2.5, Y + 2.5], outline=(90, 90, 90))
        d.rectangle([0, 0, W * Z - 1, 13], fill=(0, 0, 0)); d.text((2, 2), f"{nv}/8 visible", font=fnt(10), fill=(240, 240, 240))
        sheet.paste(im, (pad + (i % cols) * (W * Z + pad), pad + (i // cols) * (W * Z + pad)))
    sheet.save(os.path.join(HERE, "kp_training_samples.png")); print("wrote kp_training_samples.png")

if __name__ == "__main__":
    _, _, canon = kp.load(); m = _model()
    feature_slices(); features_montage(m, canon); detect_montage(m, canon); training_samples()
