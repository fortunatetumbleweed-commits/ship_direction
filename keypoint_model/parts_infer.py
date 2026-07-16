"""
Segment a frame into ship parts, recover the heading by matching the part masks to the
canonical part-layout, and reconstruct the ship. The predicted part segmentation is the
interpretable output (which parts it saw, and where).

Run:  python parts_infer.py IMG [IMG ...] [--out-dir out]
      python parts_infer.py --validate
"""
import os, glob, argparse
import numpy as np
import torch
from PIL import Image
import parts, match_and_reconstruct as mr
from parts_train import SegNet, validate
from train import to_tensor, load_real, DEVICE, W

HERE = os.path.dirname(os.path.abspath(__file__))
PCOL = {1: (255, 70, 70), 2: (150, 150, 150), 3: (70, 150, 255), 4: (255, 220, 50), 5: (255, 140, 40)}

def latest_model():
    c = sorted(glob.glob(os.path.join(HERE, "part_model_v*.pt")))
    return c[-1] if c else os.path.join(HERE, "part_model.pt")

def load_model(path=None):
    path = path or latest_model()
    ckpt = torch.load(path, map_location=DEVICE)
    state, meta = (ckpt["state_dict"], ckpt.get("meta", {})) if isinstance(ckpt, dict) and "state_dict" in ckpt else (ckpt, {})
    print(f"model: {os.path.basename(path)}" + (f"  (v{meta.get('version')}, val {meta.get('val')})" if meta else ""))
    m = SegNet().to(DEVICE); m.load_state_dict(state); m.eval(); return m

@torch.no_grad()
def segment(model, img_u8):
    P = torch.softmax(model(to_tensor(img_u8[None]).to(DEVICE)), 1)[0].cpu().numpy()
    return P, P.argmax(0)                              # probs (6,W,W), label map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    model = load_model(args.model); L0 = parts.canonical_parts(); sprite, _ = mr.build_canonical()

    if args.validate:
        xr, degs, names = load_real()
        errs, hard, synth = validate(model, xr, degs, names, L0)
        for e, n in sorted(zip(errs, names), key=lambda z: -z[0]):
            print(f"{n:28s} err={e:5.1f}{'  <hard>' if n.startswith('hard') else ''}")
        print(f"\nsynth mean={synth.mean():.1f}  hard mean={hard.mean():.1f} within20={(hard<=20).sum()}/{len(hard)}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for f in args.images:
        img = np.asarray(Image.open(f).convert("RGB"))
        P, lab = segment(model, img)
        _, open_mask = parts.mr.open_mask_from_crop(img.astype(np.int32))
        heading, shift = parts.part_pose(P, L0, open_mask=open_mask)
        seen = [parts.PART_NAMES[p] for p in range(1, 6) if (lab == p).sum() >= 8]
        # panel: predicted parts | reconstruction
        seg = img // 3
        for p in range(1, 6): seg[lab == p] = PCOL[p]
        ship = np.roll(np.roll(np.asarray(mr.render(Image.fromarray(sprite, "RGBA"), heading)), shift[0], 0), shift[1], 1)
        a = ship[..., 3:4] / 255.0; rec = (img * (1 - a) + ship[..., :3] * a).astype(np.uint8)
        Image.fromarray(np.concatenate([seg, rec], 1)).save(
            os.path.join(args.out_dir, os.path.splitext(os.path.basename(f))[0] + "_parts.png"))
        print(f"{os.path.basename(f):28s} heading={heading:6.1f}  parts seen: {', '.join(seen)}")

if __name__ == "__main__":
    main()
