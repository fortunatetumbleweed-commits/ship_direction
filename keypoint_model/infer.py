"""
Detect ship keypoints with the trained U-Net, solve the pose, and reconstruct the
ship at that heading. Each detected keypoint carries a confidence, so the output
also says which features it actually saw (graceful degradation under occlusion).

Run:  python infer.py IMG [IMG ...] [--out-dir out]
      python infer.py --validate            # heading error vs truthNNN (via keypoints)
"""
import os, glob, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
import keypoints as kp
import match_and_reconstruct as mr
from train import KeypointNet, to_tensor, detect, load_real, validate, circ_err, DEVICE, W

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"bow": (255, 60, 60), "stern_l": (80, 160, 255), "stern_r": (80, 220, 255),
          "fin_up_l": (255, 230, 40), "fin_up_r": (255, 170, 40),
          "fin_lo_l": (150, 255, 80), "fin_lo_r": (60, 210, 120), "center": (230, 230, 230)}

def latest_model():
    c = sorted(glob.glob(os.path.join(HERE, "kp_model_v*.pt")))
    return c[-1] if c else os.path.join(HERE, "kp_model.pt")

def load_model(path=None):
    path = path or latest_model()
    ckpt = torch.load(path, map_location=DEVICE)
    state, meta = (ckpt["state_dict"], ckpt.get("meta", {})) if isinstance(ckpt, dict) and "state_dict" in ckpt else (ckpt, {})
    print(f"model: {os.path.basename(path)}" + (f"  (v{meta.get('version')}, val {meta.get('val')})" if meta else ""))
    m = KeypointNet().to(DEVICE); m.load_state_dict(state); m.eval()
    return m

def reconstruct(sprite, img_u8, heading, shift, det, conf, conf_th=0.3):
    """Canonical ship stamped at the solved pose, with detected keypoints marked."""
    ship = np.roll(np.roll(np.asarray(mr.render(Image.fromarray(sprite, "RGBA"), heading)), shift[0], 0), shift[1], 1)
    base = Image.fromarray(img_u8, "RGB").convert("RGBA")
    over = Image.fromarray(ship.astype(np.uint8), "RGBA")
    out = Image.alpha_composite(base, over).convert("RGB"); d = ImageDraw.Draw(out)
    for nm, (x, y) in det.items():
        if conf[nm] >= conf_th:
            d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=COLORS[nm])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    model = load_model(args.model)
    _, _, canon = kp.load(); sprite, _ = mr.build_canonical()

    if args.validate:
        xr, degs, names = load_real()
        errs, hard, synth = validate(model, xr, degs, names, canon)
        for e, n in sorted(zip(errs, names), key=lambda z: -z[0]):
            print(f"{n:28s} err={e:5.1f}{'  <hard>' if n.startswith('hard') else ''}")
        print(f"\nsynth mean={synth.mean():.1f}  hard mean={hard.mean():.1f} within20={(hard<=20).sum()}/{len(hard)}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for f in args.images:
        img = np.asarray(Image.open(f).convert("RGB"))
        det, conf = detect(model, to_tensor(img[None]))[0]
        r = kp.solve_pose(det, conf, canon)
        seen = [n for n in kp.KP_NAMES if conf[n] >= 0.3]
        if r is None:
            print(f"{os.path.basename(f):28s} no pose ({len(seen)} kps seen)"); continue
        heading, shift, nused = r
        reconstruct(sprite, img, heading, shift, det, conf).save(
            os.path.join(args.out_dir, os.path.splitext(os.path.basename(f))[0] + "_kp.png"))
        print(f"{os.path.basename(f):28s} heading={heading:6.1f}  seen: {', '.join(seen)}")

if __name__ == "__main__":
    main()
