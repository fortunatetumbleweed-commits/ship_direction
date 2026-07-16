# Keypoint detector (part-based heading) — prototype

Detect the ship's distinctive **parts** (bow, stern corners, upper/lower sail-fin tips,
center), then solve the heading from whichever parts survive the occlusion. This is the
part-based alternative to matching the whole ship or regressing one heading number:
each feature independently constrains direction, and every detection carries a
confidence, so the system self-reports *what it saw*.

```
image → U-Net → 8 keypoint heatmaps + visibility → confident keypoints
      → solve_pose (2-D rigid fit) → heading → stamp canonical ship
```

## Status: promising concept, first model underperforms — needs work

Honest results on the 21 real labelled `heading_v9d` images:

| | keypoint model `v1` | (for comparison) template matcher | heading CNN |
|---|---|---|---|
| clean `synth_*` | **1.2°** | ~0–1° | 2.2° |
| occluded `hard_*` | **35.9°, 4/9** within 20° | 7/9 | **5.3°, 9/9** |

The building blocks are verified and strong:
- **Pose solver is exact** — 0.00° from all keypoints, ~1.3° median from just 2 noisy ones.
- **Clean-image detection is excellent** — 1.2°, so the model localizes parts well when it sees them.

But it **generalizes poorly to the real occluded frames** (35.9° vs the CNN's 5.3°). Two
causes, both visible in the training log and detections:
1. **Domain gap** — on real frames the detector places keypoints *on the occluders*
   (portrait, "Village" text, diamonds); the visibility head fails to suppress them
   because the synthetic occluders don't match the real ones closely enough. Precise
   localization is far more sensitive to this gap than the CNN's holistic heading readout.
2. **Overfitting** — `synth` error stayed ~1° while `hard` error *worsened* after epoch 6
   (best model is early). The net memorized synthetic keypoint appearance.

So the architecture is sound (and it's the most interpretable of the three — you can see
which parts it found), but as a heading estimator this first cut does **not** beat the CNN.

## To make it competitive
- **Close the domain gap** (the main lever): stronger appearance augmentation (color/blur),
  a wider/real occluder library, and — best — mix in real `(frame, keypoints)` exports from
  the game (keypoint labels are cheap: you know where each part projects).
- **Regularize** — dropout + weight decay + early stopping (v1 already overfits by ep ~6).
- **Reject occluder detections** — supervise the visibility head harder, or add a
  "background/occluder" class so keypoints on non-ship pixels are pushed down.

## Files
| file | role |
|---|---|
| `keypoints.py`  | 8 canonical keypoints + projection + `solve_pose` (all verified) |
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
