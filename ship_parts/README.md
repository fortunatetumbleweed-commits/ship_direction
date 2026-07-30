# ship_parts — standalone heading estimator (drop-in)

The region-parts model packaged for embedding in another Python app. Same weights and
same math as `../keypoint_model/`, with the training code and the dataset dependency
stripped out.

## Install

Copy the `ship_parts/` folder into your project. That's it — 4 files:

| file | size | what |
|---|---|---|
| `estimator.py` | 9 KB | the whole implementation (net + geometry) |
| `__init__.py` | — | exports `ShipHeading` |
| `part_model_v1.pt` | 1.9 MB | trained U-Net weights |
| `canonical.npz` | 2.3 KB | frozen canonical part-map + ship sprite |

Requires `numpy`, `pillow`, `torch`. **No dataset folder, no training modules.**

## Use

```python
import numpy as np
from PIL import Image
from ship_parts import ShipHeading

est = ShipHeading()                       # load once; warms the rotation cache
img = np.asarray(Image.open("frame.png").convert("RGB"))   # uint8 (80,80,3) RGB

r = est(img)
r.heading      # float degrees, 0 = up/north, clockwise
r.shift        # (dy, dx) placement of the ship
r.parts_seen   # e.g. ['bow'] or ['bow','hull','stern','sail_l','sail_r']
r.labels       # (80,80) uint8 per-pixel part ids
r.probs        # (6,80,80) float32 class probabilities

recon = est.reconstruct(img, r)           # (80,80,3) uint8, whole ship painted in
```

Options: `ShipHeading(device="cpu", step=3, model_path=..., canonical_path=...)`.

## Input contract (get this exact, or accuracy degrades silently)

- **80×80**, `uint8`, **RGB** channel order — if you're using OpenCV, convert from BGR.
- Pass the **raw frame**; the module does its own normalization (`/255 − 0.5`) internally.
- The ship icon is assumed at the game's fixed scale, roughly centered.

`estimate()` raises `ValueError` on a wrong shape, but it cannot detect BGR or a wrong
scale — those just quietly produce worse headings.

## Performance (M-series, CPU)

```
init (incl. rotation cache): ~60 ms      <- once, at startup
per frame:                   ~39 ms      <- ~25 fps
  of which net forward:      ~10 ms
  of which pose fit:         ~29 ms
```

**Device:** defaults to `cpu` on purpose. For a single 80×80 frame the MPS transfer and
launch overhead typically exceeds the compute saved — measure before switching.

**`step`** (rotation granularity, degrees) is the speed lever: `step=6` → ~27 ms,
`step=10` → ~22 ms. All tested values stay 9/9 within 20° on the real set; the small
mean-error differences between them are **noise on a 9-image sample**, not evidence that
one is more accurate. Coarser `step` does add heading quantization (max error `step/2`).

**Threading:** each `ShipHeading` instance owns its rotation cache, so instances are
independent — but don't share one instance across threads without a lock.

## Verify the extraction

```bash
python ship_parts/test_smoke.py
```

Runs the 21 labelled frames and asserts the standalone module still reproduces the
in-repo numbers (clean 0.7°, occluded 9/9 within 20°, mean 6.1°). This is the only place
the dataset is touched — it is never needed at inference time. Re-run it after changing
anything in `estimator.py`.

## What was changed in extraction (and why)

1. **Froze the canonical assets.** `parts.canonical_parts()` rebuilt the ship sprite by
   reading the 12 `synth_*.png` files *at runtime*; that's now baked into `canonical.npz`,
   so the dataset isn't a deployment dependency.
2. **Lifted `SegNet` out of `parts_train.py`**, which imported `datagen` (fonts,
   backgrounds, portrait assets) purely for training.
3. **Pre-rotation cache is explicit** (`_Rot`) instead of module-level dicts, so instances
   don't share hidden global state.

The math is unchanged — `part_pose` keeps the same three terms and the same
`lam=0.4, beta=1.0`, which is what `test_smoke.py` pins.
