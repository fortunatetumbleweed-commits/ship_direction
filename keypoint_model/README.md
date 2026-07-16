# Keypoint detector (part-based heading) — prototype

Detect the ship's distinctive **parts** (bow, stern corners, upper/lower sail-fin tips,
center), then solve the heading from whichever parts survive the occlusion. This is the
part-based alternative to matching the whole ship or regressing one heading number:
each feature independently constrains direction, and every detection carries a
confidence, so the system self-reports *what it saw*.

```
image → U-Net → 8 keypoint heatmaps + visibility
      → constellation_pose (fit the rigid part-constellation to ALL heatmaps at once)
      → heading → stamp canonical ship
```

## Status: competitive with the template matcher; interpretable

Results on the 21 real labelled `heading_v9d` images:

| | keypoint `v1` | (for comparison) template matcher | heading CNN |
|---|---|---|---|
| clean `synth_*` | **1.0°** | ~0–1° | 2.2° |
| occluded `hard_*` | **17.7°, 7/9** within 20° | 7/9 | **5.3°, 9/9** |

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
- **Denser parts** — the single-pixel keypoints are locally ambiguous; predicting small part
  *regions* (or dense canonical-coordinate votes) would give more distinctive shape.

## Files
| file | role |
|---|---|
| `keypoints.py`  | 8 canonical keypoints + projection + `solve_pose` + `constellation_pose` |
| `datagen.py`    | synthetic frames with per-keypoint (x,y) + visibility; heatmap targets |
| `train.py`      | U-Net (heatmaps + visibility); validates via detect→solve→heading |
| `infer.py`      | detect + solve + reconstruct; reports which features it saw |
| `kp_model_v1.pt`| trained weights (self-describing checkpoint) |
| `keypoints.json`| canonical keypoint coords |

## Usage
```bash
source ../.venv/bin/activate && pip install torch numpy pillow
python keypoints.py                     # (re)build canonical keypoints + preview
python datagen.py                       # labelled sample sheet to eyeball
python train.py --n-train 20000 --epochs 22   # ~5 min MPS; saves kp_model_v{VERSION}.pt
python infer.py IMG --out-dir out       # detect + solve + reconstruct
python infer.py --validate              # heading error via keypoints on the real set
```
