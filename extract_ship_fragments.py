#!/usr/bin/env python3
"""
Extract the visible ship-green fragment from ship images (e.g. the occluded hard_*
frames) and save each as a transparent-background RGBA cutout.

The ship icon is bright, saturated green; the strict mask below keeps those pixels
while rejecting duller grass-green, the yellow NPC portrait, white text/icons, and
terrain. Each fragment is saved at the SAME size as the source with pixel positions
preserved, so it overlays back onto the original frame exactly.

Examples
--------
  # all hard_* frames in the dataset -> ./ship_fragments/ (with a preview sheet)
  python extract_ship_fragments.py

  # specific files, custom output dir, no preview
  python extract_ship_fragments.py img1.png img2.png --out-dir frags --no-preview

  # a whole folder with a glob, and also write tight-cropped versions
  python extract_ship_fragments.py --in-dir some/dir --pattern '*.png' --crop

Needs: numpy, pillow  (use the project venv:  ../.venv/bin/python extract_ship_fragments.py)
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "heading_v9d_raw_images")
DEFAULT_OUT = os.path.join(HERE, "ship_fragments")

def ship_mask(a, loose=False):
    """Boolean mask of ship-green pixels. strict (default) rejects grass; loose keeps more."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if loose:
        return (g > 90) & (g > r + 30) & (g > b + 30)
    return (g > 110) & (g - r > 60) & (g - b > 50)

def extract(path, mask_fn):
    """Return (rgba_uint8, mask_bool) — ship pixels in original color, transparent elsewhere."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
    m = mask_fn(a)
    rgba = np.dstack([a.astype(np.uint8), (m * 255).astype(np.uint8)])
    return rgba, m

def make_preview(items, out_path, scale=3):
    """Side-by-side sheet: original | fragment on a gray backdrop."""
    if not items:
        return
    W = max(im.width for im, _ in items); H = max(im.height for im, _ in items)
    pad = 6
    sheet = Image.new("RGB", (2 * W * scale + 3 * pad, len(items) * (H * scale + pad) + pad), (30, 30, 30))
    for i, (orig, frag) in enumerate(items):
        y = pad + i * (H * scale + pad)
        gray = Image.new("RGBA", frag.size, (60, 60, 60, 255))
        fg = Image.alpha_composite(gray, frag).convert("RGB")
        sheet.paste(orig.resize((orig.width * scale, orig.height * scale), Image.NEAREST), (pad, y))
        sheet.paste(fg.resize((fg.width * scale, fg.height * scale), Image.NEAREST), (2 * pad + W * scale, y))
    sheet.save(out_path)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files (overrides --in-dir/--pattern)")
    ap.add_argument("--in-dir", default=DEFAULT_IN, help=f"input folder (default: {DEFAULT_IN})")
    ap.add_argument("--pattern", default="hard_*.png", help="glob within --in-dir (default: hard_*.png)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help=f"output folder (default: {DEFAULT_OUT})")
    ap.add_argument("--loose", action="store_true", help="looser green mask (keeps more, risks grass)")
    ap.add_argument("--crop", action="store_true", help="also save tight-cropped *_crop.png")
    ap.add_argument("--no-preview", action="store_true", help="skip the _preview.png sheet")
    args = ap.parse_args()

    files = args.images or sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not files:
        sys.exit(f"no images found ({args.images or os.path.join(args.in_dir, args.pattern)})")
    os.makedirs(args.out_dir, exist_ok=True)
    mask_fn = lambda a: ship_mask(a, loose=args.loose)

    preview = []
    for f in files:
        rgba, m = extract(f, mask_fn)
        name = os.path.splitext(os.path.basename(f))[0]
        n = int(m.sum())
        Image.fromarray(rgba, "RGBA").save(os.path.join(args.out_dir, name + "_fragment.png"))
        if args.crop and n:
            ys, xs = np.where(m)
            crop = rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]
            Image.fromarray(crop, "RGBA").save(os.path.join(args.out_dir, name + "_crop.png"))
        preview.append((Image.open(f).convert("RGB"), Image.fromarray(rgba, "RGBA")))
        print(f"{name:28s} {n:5d} ship px")

    if not args.no_preview:
        make_preview(preview, os.path.join(args.out_dir, "_preview.png"))
    print(f"\nsaved {len(files)} fragment(s) to {args.out_dir}")

if __name__ == "__main__":
    main()
