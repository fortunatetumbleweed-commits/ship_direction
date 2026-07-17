"""
Regenerate parts_features.png -- the "how the region-parts matching works" diagnostic:
a per-case panel with three columns

    input frame | predicted parts (segmentation) | reconstruction (truth / pred / err)

over a fixed set of clean and occluded frames, ending on t083 (whose label was corrected
210 -> 174). Companion to parts_preview.png (which shows how the parts are *defined*).

Run:  python parts_viz.py
"""
import os, re
import numpy as np
from PIL import Image, ImageDraw
import parts, match_and_reconstruct as mr
from parts_infer import load_model, segment, PCOL
from train import circ_err

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "heading_v9d_raw_images")
W, Z, PAD, HDR = mr.W, 5, 4, 18

# 2 clean + the 5 occluded cases that tell the story (light -> the corrected heavy one)
CASES = [
    ("synth_s02_truth205.png", "synth clean"),
    ("synth_s01_truth99.png",  "synth clean"),
    ("hard_t557_truth347.png", "t557 light occ"),
    ("hard_t104_truth207.png", "t104 occ"),
    ("hard_t556_truth350.png", "t556 occ (was the flip)"),
    ("hard_t082_truth168.png", "t082 occ (stern only, scene fallback)"),
    ("hard_t083_truth174.png", "t083 heavy occ (label corrected 210->174)"),
]

def truth_of(name):
    return int(re.search(r"truth(\d+)", name).group(1))

def tiles(model, L0, sprite, path):
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)
    P, lab = segment(model, img)
    _, open_mask = parts.mr.open_mask_from_crop(img.astype(np.int32))
    heading, shift = parts.part_pose(P, L0, open_mask=open_mask, ship_mask=parts.ship_green(img))
    seen = [parts.PART_NAMES[p] for p in range(1, 6) if (lab == p).sum() >= 8]
    seg = img // 3
    for p in range(1, 6): seg[lab == p] = PCOL[p]
    ship = np.roll(np.roll(np.asarray(mr.render(Image.fromarray(sprite, "RGBA"), heading)), shift[0], 0), shift[1], 1)
    a = ship[..., 3:4] / 255.0
    rec = (img * (1 - a) + ship[..., :3] * a).astype(np.uint8)
    return img, seg, rec, heading, seen

def main():
    model = load_model(); L0 = parts.canonical_parts(); sprite, _ = mr.build_canonical()
    tile = W * Z
    sheetW = 3 * tile + 4 * PAD
    sheetH = HDR + len(CASES) * (tile + PAD) + PAD
    sheet = Image.new("RGB", (sheetW, sheetH), (16, 16, 16)); d = ImageDraw.Draw(sheet)
    for c, h in enumerate(["input", "predicted parts", "reconstruction"]):
        d.text((PAD + c * (tile + PAD) + 3, 3), h, fill=(235, 235, 235))

    def cap(x, y, s):
        d.rectangle([x, y, x + len(s) * 6 + 4, y + 12], fill=(0, 0, 0))
        d.text((x + 2, y), s, fill=(240, 240, 120))

    for r, (fn, label) in enumerate(CASES):
        path = os.path.join(RAW, fn); truth = truth_of(fn)
        img, seg, rec, heading, seen = tiles(model, L0, sprite, path)
        err = circ_err(heading, truth)
        y = HDR + r * (tile + PAD)
        for c, arr in enumerate((img, seg, rec)):
            x = PAD + c * (tile + PAD)
            sheet.paste(Image.fromarray(arr).resize((tile, tile), Image.NEAREST), (x, y))
        cap(PAD + 2, y + 2, label)
        cap(PAD + (tile + PAD) + 2, y + 2, "parts: " + (",".join(seen) if seen else "-"))
        cap(PAD + 2 * (tile + PAD) + 2, y + 2, f"truth {truth}  pred {heading:.0f}  e{err:.0f}")
        print(f"{fn:28s} truth={truth} pred={heading:6.1f} err={err:4.1f} parts={seen}")

    out = os.path.join(HERE, "parts_features.png")
    sheet.save(out); print(f"\nwrote {os.path.basename(out)}")

if __name__ == "__main__":
    main()
