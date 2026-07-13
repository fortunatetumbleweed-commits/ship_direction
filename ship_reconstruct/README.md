# Ship heading estimation + de-occlusion

Reconstructs the full ship icon from an 80×80 image where the ship (bright-green
top-down sprite) may be partly hidden by map/UI overlays — **without knowing the
heading in advance**. It estimates the heading, rebuilds the whole ship at that
rotation, and reports a confidence so you know when a frame is too occluded to trust.

Runs locally on CPU. Only dependencies: `numpy`, `pillow`. **No training, no GPU.**

## Why this instead of a trained model

The clean `synth_*` images are one ship sprite rendered at known headings, so we
have a *perfect* template. Heading is then just: rotate the template through 360°,
register it onto the visible ship pixels, keep the best match. This beats a neural
net here because (a) 21 images is far too few to train on, and (b) the hard cases
are **information-limited** — when only ~30 ship pixels survive, the shape no longer
distinguishes bow from stern, so *no* method can recover the heading reliably. This
tool detects that situation and flags it instead of guessing with false confidence.

### Axis vs. direction

The ship is **left/right symmetric but front/back asymmetric**. So heading splits into:

- **Axis** (0–180°) — which way the hull lines up. Robust; recoverable from a small
  fragment because left/right carries no heading information.
- **Direction** (bow vs. stern) — a binary that needs the front/back difference to be
  visible. This is the fragile part under heavy occlusion.

The tool reports **both, with separate confidences**. That way a frame that's a clean
bow/stern flip (e.g. `t556`) still yields a correct, confident **axis** — it just warns
that the 180°-flip is possible — instead of being discarded as a total failure.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pillow
```

## Usage

```bash
# 1) Build the reusable template once from the clean synth_* images:
python reconstruct.py build-template --synth-dir ../heading_v9d_raw_images --out template.png

# 2) Reconstruct any image(s) — heading is estimated automatically:
python reconstruct.py run path/to/img.png --template template.png --out-dir out/

# 3) For a frame flagged LOW CONFIDENCE, force the heading yourself:
python reconstruct.py run path/to/img.png --template template.png --angle 180 --out-dir out/

# 4) Accuracy check against ground-truth `truthNNN` in filenames:
python reconstruct.py validate ../heading_v9d_raw_images/*.png --template template.png
```

For each input `foo.png`, `run` writes:
- `foo_ship.png`  — clean reconstructed ship sprite (transparent background) at the estimated heading
- `foo_deocc.png` — the original image with the complete ship painted over the occluders

Console output gives `heading` and `axis` (degrees, **0 = up/north, clockwise**),
`visible_px`, and a note. A frame is called out when the **axis** is too occluded to
trust, or when the axis is solid but the **bow/stern** call is uncertain (it names the
possible 180°-flip). Set `--angle` when you know the true heading.

## Accuracy (validated on the 21 labelled images)

| set | result |
|---|---|
| clean `synth_*` (12) | mean error **0.0°**, max 1° |
| **axis**, where axis-confident (19 imgs) | **mean 1.9°, max 11°** |
| **full heading**, where fully confident (18 imgs) | **mean 1.9°, max 11°** |

Zero false-confident errors. `t556` (a bow/stern flip) is reported with a correct,
confident axis and a flip warning. Only `t082` (38 px) and `t083` (29 px) — too
occluded to fix even the axis — are flagged fully low-confidence.

## How it works

1. **Template** — de-rotate every clean `synth_*` ship back to heading 0, take the
   median, and fold it left/right (the ship is L/R symmetric) → one clean canonical
   footprint (`template.png`).
2. **Segment** — isolate bright, saturated ship-green (rejects duller grassy green).
3. **Match** — rotate the template through 0–359° (3° coarse, then 1° refine); at each
   angle slide it over the visible ship pixels and score
   `coverage × fill^0.3` (how much of the visible ship it explains × how tightly the
   fragment fills it). Best (angle, shift) wins.
4. **Confidence** — reported separately for axis and direction. *Axis* confidence uses
   visible pixels, match quality, and the margin over the best *different-axis* pose.
   *Direction* confidence is the margin over the best *opposite* (bow/stern-flipped)
   pose. Small margins ⇒ ambiguous ⇒ flagged.
5. **Reconstruct** — paint the canonical ship at the winning pose.

## Tunables

- `--angle N` — force the heading (bypasses estimation; still registers position).
- `--conf-th` — confidence threshold for the LOW-CONFIDENCE flag (default 0.4).
- Segmentation thresholds live in `ship_mask()`; the ship-green color used for the
  reconstructed sprite is in `render_sprite()`.
