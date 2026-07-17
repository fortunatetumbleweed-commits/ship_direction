# Ship heading model (learned, occlusion-robust)

Estimates ship heading (0–359°, **0 = up/north, clockwise**) from an 80×80 game
frame where the ship may be almost entirely hidden, then reconstructs the full ship
at that heading. Unlike the template matcher in `../ship_reconstruct/`, this learns
**scene context** — e.g. "a hull tip poking out from under a portrait, water above it,
so the body continues under the icon" — which is what's needed when the visible
fragment alone doesn't determine the heading.

CPU/Apple-GPU (MPS), no cloud. Deps: `torch`, `numpy`, `pillow`.

## Why a model here (and not for the template matcher)

The occluded cases need world knowledge a template match can't provide, and — crucially
— the game can generate **unlimited labelled data**. We render the clean ship at a known
heading and composite the game's real occluders on top, so the net learns to read the
scene. (For lightly-occluded frames the template matcher is still simpler and exact; this
model is for the hard, context-dependent cases.)

## Results (validation on the 21 real labelled images)

| set | mean err | within 20° |
|---|---|---|
| clean `synth_*` (12) | **2.2°** | 12/12 |
| occluded `hard_*` (9) | 9.3° | **8/9** |

Per hard image: t542 1.0, t557 1.1, t390 2.0, t541 2.4, t084 6.2, t556 7.0, t104 10.1,
**t082 16.5**, **t083 37.3**. The `t082` portrait-tip case — 91° error with crude occluders,
unrecoverable for the template matcher — is now within 20° because the occluders are
realistic.

> **t083 caveat.** This model predicts 211° on t083, which originally scored 1.3° against the
> dataset's `truth210` label — but that label was **wrong** (210° puts the ship on open water;
> the true heading is ~174°, caught by `../keypoint_model/`). The label is now corrected to
> 174°, against which this model is 37° off. So this CNN's real occluded score is **8/9**, not
> the 9/9 the bad label implied — a good reminder that a black-box regressor agreeing with a
> label is not the same as being right.

> **Honest caveat.** These 9 images are the only real labelled data, so they serve as
> validation *and* their backgrounds + the portrait asset were harvested to build the
> training occluders. Expect some optimism. The headings themselves are never used in
> training, and 9/9 across varied occluders is a strong signal — but the true test is
> **fresh game frames**. Retrain with game exports (below) for production numbers.

## Files

| file | role |
|---|---|
| `datagen.py`   | synthetic data: clean ship at random heading + real occluders (portrait, village text, minimap icons, terrain cover) on purity-filtered terrain backgrounds |
| `train.py`     | small CNN regressing (sin, cos) of heading; validates on the real images each epoch |
| `infer.py`     | predict heading + reconstruct the de-occluded ship |
| `ship0.png`    | canonical ship sprite (heading 0), built from `synth_*` |
| `portrait.png` | NPC portrait occluder, extracted from `hard_t084` |
| `model_v1.pt`  | trained weights (self-describing checkpoint, see below) |

## Model versioning

Checkpoints are saved as `model_v{VERSION}.pt` and carry a `meta` block inside the file:
version, training timestamp, config (`n_train`/`epochs`/`batch`/`seed`), a hash of the
training code (`datagen.py` + `train.py`), and the validation metrics. So any checkpoint
self-identifies — `infer.py` prints the card on load and defaults to the newest `model_v*.pt`.
Bump `MODEL_VERSION` in `train.py` (or pass `--version`) when you change the training setup.

Current `model_v1`: `hard_mean 9.3° · within20 8/9 · synth_mean 2.2°` (vs corrected labels).

## Usage

```bash
source ../.venv/bin/activate          # torch, numpy, pillow

python datagen.py                     # regenerate ship0.png + a sample_sheet.png to eyeball
python train.py --n-train 24000 --epochs 25    # ~4 min on MPS; saves model_v1.pt
python train.py --version 2                     # next training iteration -> model_v2.pt
python infer.py IMG [IMG ...] --out-dir out    # newest model_v*.pt; writes *_deocc.png
python infer.py --model model_v1.pt --validate # pin a version; error vs truthNNN
```

## Closing the domain gap with real game data

Accuracy on real frames is capped by how well `datagen.py` mimics the game. To improve:

1. **Add occluder assets** — drop more `portrait*.png` cutouts (any icon that overlays
   ships) into this folder; they're picked up automatically. Extend `NAMES`/text style
   and `_house_sprite`/`_diamond_sprite` to match your UI.
2. **Better backgrounds** — point `RAW_DIR` at a folder of clean terrain crops, or export
   ship-free map tiles from the game.
3. **Best: train on real exports.** If the game can dump `(frame, heading)` pairs, mix
   them into training (they can be loaded exactly like the synthetic batch in `train.py`).
   Real frames remove the domain gap entirely.

## How it works

1. **Ship sprite** — median of de-rotated `synth_*`, L/R-folded (the ship is L/R symmetric).
2. **Synthetic frame** — ship at a random heading on a terrain background, with real
   occluders composited on top; a minimum visible hull is guaranteed (never a fully hidden
   ship, which would be a label with no signal).
3. **Net** — 4 conv blocks → global pool → FC → unit (sin, cos); heading = atan2. Circular
   by construction, so 359° and 0° are neighbours.
4. **Reconstruct** — paint the clean ship at the predicted heading over the frame.
