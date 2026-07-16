# Part-based heading (keypoints → regions)

Detect the ship's distinctive **parts**, then recover the heading by fitting the rigid
part-layout to what's visible. Two representations live here, in order of how they evolved:

1. **Point keypoints** (`train.py`, `kp_model_v1.pt`) — 8 point features + a constellation fit.
2. **Region parts** (`parts_train.py`, `part_model_v1.pt`) — the ship carved into 5 whole
   regions (bow / hull / stern / sail_l / sail_r) via per-pixel segmentation, then a dense
   part-mask fit. **This is the better one** — regions are distinctive where points are not.

```
region model:  image → U-Net → 6-class part segmentation
             → part_pose (rigid pose whose canonical part-layout best matches the masks)
             → heading → stamp canonical ship
```

## Status: region parts are the best part-based approach; interpretable

Results on the 21 real labelled `heading_v9d` images:

| | **region parts** `v1` | point keypoints `v1` | (compare) matcher | heading CNN |
|---|---|---|---|---|
| clean `synth_*` | **0.8°** | 1.0° | ~0–1° | 2.2° |
| occluded `hard_*` | **9.3°, 8/9** | 17.7°, 7/9 | 7/9 | **5.3°, 9/9** |

### Why regions beat points
A single-pixel keypoint is *locally ambiguous* — a bow tip, a sail tip and a stern corner all
look like "a green protrusion", so the point model confused them and needed the geometry fit
just to be usable (and still overfit: `hard` error drifted up after epoch 6). A **region**
carries distinctive shape and orientation: the stern is a big solid block, the bow a wedge —
they can't be confused. The segmentation model is **stable** (no overfitting) and fixes the
bow/stern flip (**t556** 10°) that beat every other method. Only **t082** misses (36° — the
ship is almost entirely behind the portrait, so only the stern nub shows; one part can't fix
orientation). This is the culmination of the project: learned occlusion-robust *features* +
rigid *part geometry*, at the region level.

---

### (History) point-keypoint model — the constellation lesson

### The key move: fit the constellation, don't assemble parts independently

The parts have only **3 degrees of freedom** (heading + x/y), not 16 — they can't be placed
independently. Taking each heatmap's argmax and solving from those points fails on real
frames: locally a bow tip, sail tip, and stern corner all look like "a green protrusion", so
the model puts, say, *bow* on a sail tip, and the pose solve gets inconsistent points
(**35.9°, 4/9**). Instead, `constellation_pose` searches for the single rigid pose whose
projected keypoint constellation best explains **all** the heatmaps at once, each part
weighted by its confidence. A spurious peak can't win unless every other part also lines up
under the same pose. Same weights, no retraining — **35.9° → 17.7°, 4/9 → 7/9** (e.g.
t082 129°→8°, t542 40°→2°, t084 51°→4°).

### Honest limits
- **Clean detection is excellent (1.0°)** and the constellation fit is robust, but 2 real
  cases still miss: t083 (a 29-px sliver — too little to detect any part) and t556 (a
  bow/stern flip the near-symmetric heatmaps resolve the wrong way).
- **Domain gap remains** — the heatmaps still leak onto real occluders the synthetic ones
  don't cover; the constellation fit tolerates it but doesn't remove it.

## To push further
- **Close the domain gap** (main lever): stronger appearance augmentation, a wider/real
  occluder library, and best — real `(frame, keypoints)` exports from the game (labels are
  cheap: you know where each part projects).
- **Reject occluder detections** — supervise the visibility head harder, or add a
  "background/occluder" class so responses on non-ship pixels go to zero.
- **Denser parts** — the single-pixel keypoints are locally ambiguous; predicting part
  *regions* gives distinctive shape. **→ done: the region-parts model above (9.3°, 8/9).**

## Files
| file | role |
|---|---|
| **region parts** | |
| `parts.py`        | 5 canonical part regions + projection + `part_pose` (dense part-mask fit) |
| `parts_train.py`  | segmentation U-Net (6-class); validates via segment→fit→heading |
| `parts_infer.py`  | segment + fit + reconstruct; shows the predicted parts |
| `part_model_v1.pt`| trained weights (self-describing checkpoint) |
| **point keypoints** | |
| `keypoints.py`    | 8 canonical keypoints + projection + `solve_pose` + `constellation_pose` |
| `datagen.py`      | synthetic frames + per-keypoint / per-part labels (shared occluder compositing) |
| `train.py`        | keypoint U-Net (heatmaps + visibility) |
| `infer.py`        | keypoint detect + solve + reconstruct |
| `viz.py`          | regenerate all diagnostic images |

## Usage
```bash
source ../.venv/bin/activate && pip install torch numpy pillow

# region parts (recommended)
python parts.py                              # canonical part regions + verify fit + preview
python parts_train.py --n-train 20000 --epochs 22   # ~6 min MPS; saves part_model_v{VERSION}.pt
python parts_infer.py IMG --out-dir out      # segment + fit + reconstruct
python parts_infer.py --validate             # heading error on the real set

# point keypoints (earlier version)
python train.py --n-train 20000 --epochs 22 ; python infer.py --validate
python viz.py                                # regenerate diagnostics
```
