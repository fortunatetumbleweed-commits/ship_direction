"""
Predict ship heading from an 80x80 image with the trained CNN, and reconstruct the
de-occluded ship at that heading.

Run:  python infer.py IMG [IMG ...] [--out-dir out]
      python infer.py --validate            # error vs truthNNN on the raw images
"""
import os, re, glob, argparse
import numpy as np
import torch
from PIL import Image
import datagen as dg
from train import HeadingNet, to_tensor, vec_to_deg, circ_err, load_real, DEVICE

HERE = os.path.dirname(os.path.abspath(__file__))
W = 80; CENTER = (40.0, 40.0)

def latest_model():
    """Newest model_v*.pt (fallback to legacy model.pt)."""
    cands = sorted(glob.glob(os.path.join(HERE, "model_v*.pt")))
    return cands[-1] if cands else os.path.join(HERE, "model.pt")

def load_model(path=None):
    path = path or latest_model()
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:      # self-describing checkpoint
        state, meta = ckpt["state_dict"], ckpt.get("meta", {})
        print(f"model: {os.path.basename(path)}"
              + (f"  (v{meta.get('version')}, trained {meta.get('trained_at')}, "
                 f"val {meta.get('val')})" if meta else ""))
    else:                                                    # legacy raw state_dict
        state = ckpt; print(f"model: {os.path.basename(path)} (legacy, no metadata)")
    m = HeadingNet().to(DEVICE); m.load_state_dict(state); m.eval()
    return m

@torch.no_grad()
def predict(model, img_u8):
    x = to_tensor(img_u8[None])
    return float(vec_to_deg(model(x.to(DEVICE)).cpu().numpy())[0])

def reconstruct(sprite, img_u8, heading):
    """Overlay the clean ship at the predicted heading (centered) onto the image."""
    ship = dg.render_ship(sprite, heading)
    base = Image.fromarray(img_u8, "RGB").convert("RGBA")
    over = Image.fromarray(ship.astype(np.uint8), "RGBA")
    return Image.alpha_composite(base, over).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--model", default=None, help="checkpoint path (default: newest model_v*.pt)")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    model = load_model(args.model); sprite = dg.build_ship_sprite()

    if args.validate:
        xr, degs, names = load_real()
        pred = vec_to_deg(model(xr.to(DEVICE)).detach().cpu().numpy())
        err = circ_err(pred, degs)
        for e, p, t, n in sorted(zip(err, pred, degs, names), key=lambda z: -z[0]):
            mark = "" if not n.startswith("hard") else "  <hard>"
            print(f"{n:28s} truth={t:3d} pred={p:6.1f} err={e:5.1f}{mark}")
        hard = np.array([e for e, n in zip(err, names) if n.startswith("hard")])
        synth = np.array([e for e, n in zip(err, names) if n.startswith("synth")])
        print(f"\nsynth mean={synth.mean():.1f}  hard mean={hard.mean():.1f} "
              f"median={np.median(hard):.1f} within20={(hard<=20).sum()}/{len(hard)}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for f in args.images:
        img = np.asarray(Image.open(f).convert("RGB"))
        h = predict(model, img)
        base = os.path.splitext(os.path.basename(f))[0]
        reconstruct(sprite, img, h).save(os.path.join(args.out_dir, base + "_deocc.png"))
        print(f"{os.path.basename(f):28s} heading={h:6.1f}")

if __name__ == "__main__":
    main()
