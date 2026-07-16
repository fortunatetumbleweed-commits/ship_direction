"""
Region/part features for the ship (the successor to point keypoints). Instead of 8
tiny keypoints, the ship is carved into 5 whole parts, each a distinctive region:
  bow, hull, stern, sail_l, sail_r  (+ background).

A segmentation net predicts per-pixel part probabilities; the heading is recovered by
`part_pose` -- the rigid pose whose projected canonical part-layout best matches the
predicted part masks (the dense analogue of the keypoint constellation fit).

`python parts.py` verifies the pose fit on ground-truth masks and writes a preview.
"""
import os, math
import numpy as np
from PIL import Image
import keypoints as kp
import match_and_reconstruct as mr

HERE = os.path.dirname(os.path.abspath(__file__))
W = mr.W
PART_NAMES = ["bg", "bow", "hull", "stern", "sail_l", "sail_r"]   # index 0..5
NP = 5                                                            # ship parts (1..5)
STERN_Y, SAIL_T = 44, 4.0

def canonical_parts():
    """Return an (W,W) uint8 part-label map of the canonical ship (0=bg, 1..5=parts)."""
    cmask, _, _ = kp.load()
    ys, xs = np.where(cmask); cx = xs.mean()
    Y, X = np.mgrid[0:W, 0:W]; L = np.zeros((W, W), np.uint8)
    lat = np.abs(X - cx)
    L[cmask & (Y <= 28)] = 1                                      # bow
    L[cmask & (Y >= STERN_Y)] = 3                                 # stern
    mid = cmask & (Y > 28) & (Y < STERN_Y)
    L[mid & (lat <= SAIL_T)] = 2                                  # hull
    L[mid & (lat > SAIL_T) & (X < cx)] = 4                        # sail_l
    L[mid & (lat > SAIL_T) & (X >= cx)] = 5                       # sail_r
    return L

def render_labels(L0, heading, shift=(0, 0)):
    """Project the canonical part-label map to (heading, shift) -- NEAREST keeps labels."""
    r = Image.fromarray(L0).rotate(-heading % 360, resample=Image.NEAREST, center=mr.CENTER)
    return np.roll(np.roll(np.asarray(r), shift[0], 0), shift[1], 1)

_MASK_FFT, _FOOT_FFT = {}, {}   # caches keyed by rotation step
def _mask_ffts(L0, step):
    if step in _MASK_FFT: return _MASK_FFT[step]
    L0img = Image.fromarray(L0); heads = list(range(0, 360, step)); ffts = []
    for h in heads:
        r = np.asarray(L0img.rotate(-h % 360, resample=Image.NEAREST, center=mr.CENTER))
        ffts.append([np.conj(np.fft.rfft2((r == p).astype(np.float32))) for p in range(1, 6)])
    _MASK_FFT[step] = (heads, ffts); return _MASK_FFT[step]

def _foot_ffts(L0, step):
    if step in _FOOT_FFT: return _FOOT_FFT[step]
    foot = Image.fromarray((L0 > 0).astype(np.uint8) * 255)
    ffts = [np.conj(np.fft.rfft2((np.asarray(foot.rotate(-h % 360, resample=Image.NEAREST, center=mr.CENTER)) > 127).astype(np.float32)))
            for h in range(0, 360, step)]
    _FOOT_FFT[step] = ffts; return ffts

def part_pose(P, L0, open_mask=None, lam=0.6, gate=0.25, step=3, smax=16):
    """Recover (heading, (dy,dx)) by matching predicted part probs P (6,W,W) to the
    rotated canonical part masks (sum of per-part cross-correlations).

    When `open_mask` is given AND the overlap profile is degenerate (nearly flat -- a
    single small symmetric part carries no orientation, e.g. only a stern tip visible),
    fall back to the scene: penalize poses whose hull would fall over open water. Only
    then, so poses the parts DO determine are left untouched.
    """
    heads, ffts = _mask_ffts(L0, step)
    Pf = [np.fft.rfft2(P[p]) for p in range(1, 6)]
    idxs = np.r_[0:smax + 1, W - smax:W]
    accs, ovmax = [], []
    for mf in ffts:
        acc = np.zeros((W, W), np.float32)
        for j in range(NP):
            acc += np.fft.irfft2(Pf[j] * mf[j], s=(W, W))
        accs.append(acc); ovmax.append(float(acc[np.ix_(idxs, idxs)].max()))
    ovmax = np.array(ovmax)
    degenerate = open_mask is not None and (ovmax.max() - ovmax.min()) / max(ovmax.max(), 1e-6) < gate
    if degenerate:
        Of = np.fft.rfft2(open_mask.astype(np.float32)); ff = _foot_ffts(L0, step)
        accs = [a - lam * np.maximum(np.fft.irfft2(Of * ff[k], s=(W, W)), 0) for k, a in enumerate(accs)]
    best = (-1e9, 0, (0, 0))
    for h, acc in zip(heads, accs):
        sub = acc[np.ix_(idxs, idxs)]; ij = np.unravel_index(int(sub.argmax()), sub.shape)
        dy, dx = idxs[ij[0]], idxs[ij[1]]
        if sub[ij] > best[0]:
            best = (float(sub[ij]), h, (dy - W if dy > smax else dy, dx - W if dx > smax else dx))
    return best[1], best[2]

if __name__ == "__main__":
    import random
    from PIL import ImageDraw, ImageFont
    L0 = canonical_parts()
    print("part pixel counts:", {PART_NAMES[p]: int((L0 == p).sum()) for p in range(1, 6)})

    # verify pose fit on ground-truth (one-hot) masks: exact, then with occluded parts
    random.seed(0); np.random.seed(0)
    def circ(a, b): return abs((a - b + 180) % 360 - 180)
    def onehot(L):
        P = np.zeros((6, W, W), np.float32)
        for p in range(6): P[p] = (L == p)
        return P
    ex = []
    for _ in range(200):
        h = random.randrange(0, 360); sh = (random.randint(-5, 5), random.randint(-5, 5))
        P = onehot(render_labels(L0, h, sh)); ph, _ = part_pose(P, L0, step=1)
        ex.append(circ(ph, h))
    print(f"exact recovery: max_err={max(ex)} deg")
    occ = []
    for _ in range(200):
        h = random.randrange(0, 360); P = onehot(render_labels(L0, h))
        drop = random.sample(range(1, 6), random.randint(1, 3))    # hide 1-3 parts
        for p in drop: P[p] = 0
        ph, _ = part_pose(P, L0, step=2); occ.append(circ(ph, h))
    print(f"with 1-3 parts hidden: mean_err={np.mean(occ):.2f}  median={np.median(occ):.2f}  p90={np.percentile(occ,90):.1f}")

    PCOL = {1: (255, 70, 70), 2: (150, 150, 150), 3: (70, 150, 255), 4: (255, 220, 50), 5: (255, 140, 40)}
    Z, pad, poses = 6, 6, [0, 45, 120, 210, 300]
    sheet = Image.new("RGB", (len(poses) * (W * Z + pad) + pad, W * Z + pad + 16), (18, 18, 18)); d = ImageDraw.Draw(sheet)
    for i, h in enumerate(poses):
        L = render_labels(L0, h); im = np.zeros((W, W, 3), np.uint8)
        for p in range(1, 6): im[L == p] = PCOL[p]
        sheet.paste(Image.fromarray(im).resize((W * Z, W * Z), Image.NEAREST), (pad + i * (W * Z + pad), 16))
        d.text((pad + i * (W * Z + pad) + 2, 2), f"heading {h}", fill=(230, 230, 230))
    sheet.save(os.path.join(HERE, "parts_preview.png")); print("wrote parts_preview.png")
