"""
Build a Hugging Face Hub-ready folder in hf_export/.

Converts the .pt checkpoint to safetensors (the Hub flags raw pickles), writes
config.json from the checkpoint meta, and assembles the model card, figures,
loader, and a Gradio app for a Space.

Run:  python ship_parts/export_hf.py
Then: hf auth login && hf upload <user>/ship-direction ./hf_export --repo-type=model
"""
import json, os, shutil
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "hf_export")
REPO_ID = "fortunatetumbleweed/ship-direction"      # substituted into the model card

def main():
    shutil.rmtree(OUT, ignore_errors=True)          # never publish stale files or __pycache__
    os.makedirs(os.path.join(OUT, "docs"), exist_ok=True)
    ck = torch.load(os.path.join(HERE, "part_model_v1.pt"), map_location="cpu", weights_only=False)
    meta, state = ck["meta"], ck["state_dict"]

    # safetensors requires contiguous, non-shared tensors
    save_file({k: v.contiguous().cpu() for k, v in state.items()},
              os.path.join(OUT, "model.safetensors"),
              metadata={"format": "pt", "arch": meta["arch"], "version": str(meta["version"])})

    # meta -> config.json. Drop `val_no_fallback`: those are the heading-CNN's numbers
    # (9.33 / 8-of-9), not this model's, and publishing both reads as contradictory.
    cfg = {k: v for k, v in meta.items() if k != "val_no_fallback"}
    cfg.update({"num_classes": len(meta["parts"]), "input_size": [80, 80],
                "normalization": "x/255.0 - 0.5", "channel_order": "RGB"})
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

    for src, dst in [("canonical.npz", "canonical.npz"), ("estimator.py", "estimator.py"),
                     ("docs/01-parts.png", "docs/01-parts.png"),
                     ("docs/02-pipeline.png", "docs/02-pipeline.png")]:
        shutil.copy(os.path.join(HERE, src), os.path.join(OUT, dst))
    shutil.copy(os.path.join(ROOT, "LICENSE"), os.path.join(OUT, "LICENSE"))

    for name, text in (("README.md", CARD), ("app.py", APP), ("requirements.txt", REQS)):
        open(os.path.join(OUT, name), "w").write(text.replace("{REPO_ID}", REPO_ID))

    print(f"wrote {OUT}")
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f"  {os.path.getsize(p) if os.path.isfile(p) else '-':>9} {f}")

CARD = '''---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - pose-estimation
  - occlusion
  - unet
  - synthetic-data
  - segmentation
---

# Ship direction — a U-Net that reads heading through heavy occlusion

Estimates the **heading** of a ship icon in an 80x80 top-down game frame where the ship may
be almost entirely hidden behind portraits, village-name text, or map markers, and
reconstructs the whole ship at that heading. `0 deg = up (north)`, clockwise.

The network labels every pixel as a ship **part**; a rigid geometric fit then turns those
part masks into an angle, refusing any pose that would place the hull on open water.

![the 5 parts](docs/01-parts.png)

![pipeline on real frames](docs/02-pipeline.png)

The bottom two rows are the point: with a **single part** visible — only a stern (t082),
only a bow (t083) — the geometry still pins the heading.

## Two stages

```
frame -> SegNet (U-Net) -> per-pixel part labels -> part_pose (rigid fit) -> heading
```

The checkpoint is only half the system. `SegNet` outputs `(6, 80, 80)` class probabilities
(`bg, bow, hull, stern, sail_l, sail_r`) and contains no notion of an angle. `part_pose`
(plain NumPy/FFT, in `estimator.py`) searches all rotations for the single rigid pose that
best explains those masks, scoring three terms:

| term | meaning |
|---|---|
| + part overlap | predicted parts match the canonical layout (recall on labels) |
| + green coverage | footprint covers the *actual* visible ship pixels (recall on pixels) |
| - open-water penalty | footprint must **not** sit on plainly visible water (precision) |

The last term is what keeps reconstructions physically consistent: a ship may hide *under*
an occluder, but not float on open water where it would have been seen.

## Usage

```python
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from estimator import ShipHeading   # this repo's estimator.py

REPO = "{REPO_ID}"
est = ShipHeading(
    model_path=hf_hub_download(REPO, "model.safetensors"),
    canonical_path=hf_hub_download(REPO, "canonical.npz"),
)

img = np.asarray(Image.open("frame.png").convert("RGB"))   # uint8 (80,80,3), RGB
r = est(img)
r.heading      # float degrees, 0 = up/north, clockwise
r.parts_seen   # e.g. ['bow']
r.labels       # (80,80) uint8 part ids
recon = est.reconstruct(img, r)   # (80,80,3) uint8, whole ship painted in
```

Requires `torch`, `numpy`, `pillow`, `safetensors`.

### Input contract

A wrong shape raises; these fail **silently**:

- **RGB** channel order (OpenCV gives BGR — convert first)
- **80x80 uint8**, the raw frame; normalization (`x/255 - 0.5`) happens inside
- ship at the game's fixed scale, roughly centered

## Results

On 21 labelled real frames:

| set | mean error | within 20 deg |
|---|---|---|
| clean (12) | **0.7 deg** | 12/12 |
| occluded (9) | **6.1 deg** | **9/9** |

Speed on an M-series CPU: ~60 ms init, then ~39 ms/frame (10 ms network + 29 ms pose fit).
Defaults to CPU on purpose — for a single 80x80 frame, GPU transfer overhead exceeds the
compute saved.

## Training

20,000 **synthetic** frames: the clean ship rendered at a random heading with real game
occluders composited on top. Labels come free by rotating the canonical part map, so nothing
was hand-annotated — and only *visible* parts are labelled, teaching the network to segment
what it can actually see. Selected by lowest heading error on the real frames, not by pixel
accuracy. 473k parameters, 22 epochs.

## Honest limitations

- **The 9 real occluded frames are the only real labelled data**, and their backgrounds and
  the portrait asset were harvested to build the training occluders. Expect some optimism;
  the true test is fresh frames.
- Fixed to one ship sprite at one scale. A different icon needs retraining and a new
  canonical map.
- The green-based masks (`ship_green`, `open_mask_from_crop`) are tuned to this game's
  palette and will not transfer unchanged.

## A note on the labels

One dataset frame (t083) was originally labelled 210 deg, which is physically impossible —
it places the ship on open water. This model read **174 deg**; a heading-regression CNN and a
template matcher both answered ~211 deg, i.e. they made the *same* mistake as the bad label.
The label was corrected to 174. A black-box model agreeing with a label is not evidence the
label is right.

## License and provenance

Code and weights are Apache-2.0. The canonical ship sprite in `canonical.npz` and the
validation frames are derived from a commercial game's artwork; those underlying assets are
not covered by this license and remain the property of their rights holder. Published for
research and educational use.

Source: https://github.com/fortunatetumbleweed-commits/ship_direction
'''

APP = '''"""Gradio demo for the ship-direction model (Hugging Face Space)."""
import numpy as np
import gradio as gr
from PIL import Image
from estimator import ShipHeading, PART_NAMES

est = ShipHeading()          # reads model.safetensors + canonical.npz next to this file
PCOL = {1: (255, 70, 70), 2: (155, 155, 160), 3: (70, 150, 255),
        4: (255, 220, 50), 5: (255, 140, 40)}

def run(image):
    if image is None:
        return None, None, "Upload an 80x80 frame."
    img = np.asarray(Image.fromarray(image).convert("RGB").resize((80, 80), Image.NEAREST))
    r = est(img)
    seg = img // 3
    for p in range(1, 6):
        seg[r.labels == p] = PCOL[p]
    seen = ", ".join(r.parts_seen) or "nothing"
    return (Image.fromarray(seg).resize((320, 320), Image.NEAREST),
            Image.fromarray(est.reconstruct(img, r)).resize((320, 320), Image.NEAREST),
            f"### Heading: {r.heading:.0f}deg\\n\\nParts visible: **{seen}**")

with gr.Blocks(title="Ship direction") as demo:
    gr.Markdown(
        "# Ship direction from an occluded frame\\n"
        "A U-Net segments the ship into 5 parts (bow / hull / stern / sail_l / sail_r); "
        "a rigid geometric fit turns those masks into a heading. `0deg = up`, clockwise.\\n\\n"
        "Even a **single visible part** is usually enough."
    )
    with gr.Row():
        inp = gr.Image(label="input frame (80x80)", type="numpy", height=320)
        seg = gr.Image(label="predicted parts", height=320)
        rec = gr.Image(label="reconstruction", height=320)
    out = gr.Markdown()
    inp.change(run, inp, [seg, rec, out])
    gr.Markdown(
        "Colors: <span style='color:#ff4646'>bow</span>, "
        "<span style='color:#9b9ba0'>hull</span>, "
        "<span style='color:#4696ff'>stern</span>, "
        "<span style='color:#ffdc32'>sail_l</span>, "
        "<span style='color:#ff8c28'>sail_r</span>"
    )

demo.launch()
'''

REQS = "torch\nnumpy\npillow\nsafetensors\ngradio\n"

if __name__ == "__main__":
    main()
