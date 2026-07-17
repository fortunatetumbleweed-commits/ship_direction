# Ship heading estimation & de-occlusion

Recover the **heading** of a ship icon in an 80×80 top-down game frame — where the ship
may be heavily occluded by portraits, village-name text, and minimap markers — and
reconstruct the whole ship at that heading. Heading is `0° = up (north)`, clockwise.

The repo holds three complementary approaches plus the tooling and datasets.

## Approaches

| approach | where | needs training | best for |
|---|---|---|---|
| **Template matcher** | `match_and_reconstruct.py`, `ship_reconstruct/` | no | interpretable; matches a canonical ship to the visible fragment over all 360°, returns ranked heading candidates + confidence. Adds optional occluder-consistency and shape (pointy-in-blob) terms. |
| **Learned heading CNN** | `ship_heading_model/` | yes (synthetic) | robustness under heavy occlusion — uses whole-frame scene context (water above a tip, occluder position) that the fragment alone can't provide. |
| **Part-based (keypoints → regions)** | `keypoint_model/` | yes (synthetic) | segment the ship into distinctive **parts** (bow / hull / stern / sail_l / sail_r), fit the rigid part-layout to what's visible, penalizing any hull placed on open water. Most interpretable; 0.7° clean, **9/9** on the real occluded frames (6.1°) — the only approach that read t083 correctly, catching a wrong dataset label. See its README. |

The matcher and the CNN are the reverse of each other: the matcher reasons from the
fragment's own shape (great when a feature is visible, honest when it isn't); the CNN
reads the surrounding scene (resolves the ambiguous fragments the shape can't).

## Layout

```
extract_ship_fragments.py   # extract the visible ship-green fragment as an RGBA cutout
match_and_reconstruct.py    # fragment -> ranked heading candidates + confidence + reconstruction
make_report.py              # self-contained HTML report (original / extraction / reconstruction)
ship_reconstruct/           # standalone deterministic matcher + reconstruction (+ README)
ship_heading_model/         # CNN heading regressor: datagen / train / infer, versioned model_v*.pt (+ README)
keypoint_model/             # (in progress) part/keypoint detector
heading_v9d_raw_images/     # dataset: 12 clean synth_* + 9 occluded hard_*  (80x80, truthNNN labels)
heading_hard_cases/         # dataset: 22 full 400x190 game frames + metadata.jsonl (hard real cases)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pillow          # template matcher / extraction / report
pip install torch                 # only for ship_heading_model / keypoint_model
```

## Quick start

```bash
# 1) extract visible ship fragments from the raw frames
python extract_ship_fragments.py                       # -> ship_fragments/

# 2) template-match each fragment -> heading candidates + reconstruction montage
python match_and_reconstruct.py ship_fragments/*_fragment.png --out-dir match_out

# 3) HTML report over a folder of full frames (locates ship, extracts, reconstructs, classifies)
python make_report.py --in-dir heading_hard_cases --out heading_hard_cases/report.html

# 4) learned CNN: predict heading + reconstruct, or validate against the labels
python ship_heading_model/infer.py --validate
```

## Results (validation on the 21 labelled `heading_v9d` images)

| | template matcher (shape+occluder) | learned CNN (`model_v1`) | region parts (`part_model_v1`) |
|---|---|---|---|
| clean `synth_*` (12) | ~0–1° | mean 2.2° | mean **0.7°** |
| occluded `hard_*` (9) | 7/9 within 20° | 8/9 within 20°, mean 9.3° | **9/9** within 20°, mean **6.1°** |

The template matcher labels each case **self-sufficient** (fragment shape fixes the
heading), **needs scene** (axis clear, bow/stern is a coin-flip), or **underdetermined**
(too occluded) — see `make_report.py` output. The CNN resolves the "needs scene" cases by
looking at the whole frame.

> These numbers use the **corrected** t083 label (see below). Under the original (wrong) label,
> the CNN scored 9/9 / 5.3° — but its t083 hit was agreement with a bad label, not a correct
> read. Correcting the label re-ranks the region-parts model to the top.

## The core finding

Heading recovery is an *information* question, not just an algorithm one. A fragment is
**self-sufficient** when it contains a directional feature (a pointy bow, a sail tip);
it **needs an extra bit** (which end is the bow) when only a symmetric blunt piece shows —
and that bit lives in the scene (the occluder side), not the fragment. There's a ladder of
increasingly powerful shape cues — *area → outline → oriented-outline → recognized parts* —
and the part-based detector is the principled top of that ladder: learn the parts, and
whichever survives the occlusion tells you the direction — but only if the parts are
*distinctive* (whole regions, not ambiguous points) and you enforce that they form one rigid
ship (fit the whole part-layout, don't assemble parts independently), *and* refuse to place the
hull on open water where no ship was seen. Doing all three — region segmentation + dense
part-mask fit + an open-water penalty — fixes the bow/stern flip nothing else could, keeps every
reconstruction physically consistent, and stays fully interpretable. It's correct on all 9 real
occluded frames — including **t083, which the dataset originally mislabeled** (`truth210` puts
the ship on open water). The part model read 174°; the CNN and the matcher both read ~211°,
making the *same* mistake as the label. That's the cautionary tale: a bad label survives when
your models agree with it. Reading the pixels honestly — enforce scene consistency, don't just
count matched pixels — caught it. The label is now corrected to 174° so it can't quietly poison
future validation (see `keypoint_model/README.md`).

Each sub-tool has its own README with details and honest limitations.
