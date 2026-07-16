"""
Train the keypoint detector: a small U-Net that outputs one heatmap per keypoint
plus a visibility logit per keypoint. Heatmap targets (Gaussians, zeroed where
occluded) are built on the fly per batch. Validation runs the whole pipeline on the
real labelled 80x80 images: detect keypoints -> solve_pose -> heading error.

Run:  python train.py [--n-train K] [--epochs N] [--version V]
Saves a self-describing checkpoint to kp_model_v{VERSION}.pt.
"""
import os, re, glob, time, argparse, random, hashlib, datetime
from copy import deepcopy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import datagen as kdg
import keypoints as kp

HERE = os.path.dirname(os.path.abspath(__file__))
W = kdg.W; K = kdg.K
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_VERSION = 1
RAW = os.path.join(os.path.dirname(HERE), "heading_v9d_raw_images")

def code_hash():
    h = hashlib.sha256()
    for fn in ("datagen.py", "keypoints.py", "train.py"):
        with open(os.path.join(HERE, fn), "rb") as f: h.update(f.read())
    return h.hexdigest()[:12]

def to_tensor(imgs_u8):
    x = torch.from_numpy(imgs_u8.astype(np.float32) / 255.0 - 0.5)
    return x.permute(0, 3, 1, 2).contiguous()

def circ_err(a, b): return abs((a - b + 180) % 360 - 180)

# ---------------- model ----------------
def cbr(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

class KeypointNet(nn.Module):
    def __init__(self, k=K):
        super().__init__()
        self.e1 = nn.Sequential(cbr(3, 32), cbr(32, 32))
        self.e2 = nn.Sequential(cbr(32, 64), cbr(64, 64))
        self.e3 = nn.Sequential(cbr(64, 128), cbr(128, 128))
        self.pool = nn.MaxPool2d(2)
        self.d2 = nn.Sequential(cbr(128 + 64, 64), cbr(64, 64))
        self.d1 = nn.Sequential(cbr(64 + 32, 32), cbr(32, 32))
        self.head = nn.Conv2d(32, k, 1)
        self.vis = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, k))
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        v = self.vis(e3)
        d = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.d2(torch.cat([d, e2], 1))
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.d1(torch.cat([d, e1], 1))
        return self.head(d), v

def make_heatmaps(coords, vis, sigma=2.0):
    B, k, _ = coords.shape
    ar = torch.arange(W, device=coords.device, dtype=torch.float32)
    xs, ys = ar.view(1, 1, 1, W), ar.view(1, 1, W, 1)
    cx, cy = coords[..., 0].view(B, k, 1, 1), coords[..., 1].view(B, k, 1, 1)
    H = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return H * vis.view(B, k, 1, 1)

# ---------------- data ----------------
def gen_dataset(sprite, canon, bgs, ports, n):
    X = np.empty((n, W, W, 3), np.uint8); C = np.empty((n, K, 2), np.float32); V = np.empty((n, K), np.float32)
    for i in range(n):
        img, _, kps, vis = kdg.sample(sprite, canon, bgs, ports)
        X[i] = img
        C[i] = [kps[nm] for nm in kp.KP_NAMES]
        V[i] = [vis[nm] for nm in kp.KP_NAMES]
    return X, C, V

def load_real():
    imgs, degs, names = [], [], []
    for f in sorted(glob.glob(os.path.join(RAW, "*truth*.png"))):
        imgs.append(np.asarray(Image.open(f).convert("RGB")))
        degs.append(int(re.search(r"truth(\d+)", f).group(1))); names.append(os.path.basename(f))
    return to_tensor(np.stack(imgs)), np.array(degs), names

@torch.no_grad()
def heatmaps_conf(model, x):
    """x: (B,3,80,80) -> (heatmaps (B,K,W,W) np, conf (B,K) np)."""
    heat, vislog = model(x.to(DEVICE))
    return heat.cpu().numpy(), torch.sigmoid(vislog).cpu().numpy()

@torch.no_grad()
def detect(model, x):
    """x: (B,3,80,80) -> list of (det{name:(x,y)}, conf{name:float}) per image (argmax)."""
    hm, cf = heatmaps_conf(model, x); out = []
    for b in range(x.shape[0]):
        det, cc = {}, {}
        for i, nm in enumerate(kp.KP_NAMES):
            iy, ix = np.unravel_index(int(hm[b, i].argmax()), (W, W))
            det[nm] = (float(ix), float(iy)); cc[nm] = float(cf[b, i])
        out.append((det, cc))
    return out

@torch.no_grad()
def validate(model, xr, degs, names, canon):
    """Heading via the geometry-aware constellation fit over the heatmaps."""
    model.eval(); hm, cf = heatmaps_conf(model, xr); errs = []
    for b, truth in enumerate(degs):
        heading, _ = kp.constellation_pose(hm[b], cf[b], canon)
        errs.append(circ_err(heading, truth))
    errs = np.array(errs)
    hard = np.array([e for e, n in zip(errs, names) if n.startswith("hard")])
    synth = np.array([e for e, n in zip(errs, names) if n.startswith("synth")])
    return errs, hard, synth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", default=str(MODEL_VERSION))
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"device={DEVICE}  keypoints={K}")
    sprite, canon, bgs, ports = kdg.assets()
    print(f"generating {args.n_train} samples ..."); t0 = time.time()
    X, C, V = gen_dataset(sprite, canon, bgs, ports, args.n_train)
    print(f"  done in {time.time()-t0:.0f}s")
    Xt = to_tensor(X); Ct = torch.from_numpy(C); Vt = torch.from_numpy(V)
    xr, degs, names = load_real()

    model = KeypointNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = Xt.shape[0]; best = 1e9; best_state = None
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i+args.batch]
            xb = Xt[idx].to(DEVICE); cb = Ct[idx].to(DEVICE); vb = Vt[idx].to(DEVICE)
            target = make_heatmaps(cb, vb)
            ph, pv = model(xb)
            loss = 100.0 * F.mse_loss(ph, target) + 0.5 * F.binary_cross_entropy_with_logits(pv, vb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        errs, hard, synth = validate(model, xr, degs, names, canon)
        tag = ""
        if hard.mean() < best:
            best = hard.mean(); best_state = deepcopy(model.state_dict()); tag = "  *best"
        print(f"ep{ep:02d} loss={tot/n:.4f}  synth_mean={synth.mean():5.1f}  "
              f"hard_mean={hard.mean():5.1f} hard_within20={(hard<=20).sum()}/{len(hard)}{tag}")

    model.load_state_dict(best_state)
    errs, hard, synth = validate(model, xr, degs, names, canon)
    meta = {"version": args.version, "arch": "KeypointNet", "keypoints": kp.KP_NAMES,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"), "code_hash": code_hash(),
            "args": {k_: getattr(args, k_) for k_ in ("n_train", "epochs", "batch", "seed")},
            "val": {"hard_mean": round(float(hard.mean()), 2), "hard_within20": int((hard <= 20).sum()),
                    "n_hard": int(len(hard)), "synth_mean": round(float(synth.mean()), 2)}}
    path = os.path.join(HERE, f"kp_model_v{args.version}.pt")
    torch.save({"state_dict": best_state, "meta": meta}, path)
    print(f"\nsaved {os.path.basename(path)}")
    for k_, v_ in meta.items(): print(f"  {k_}: {v_}")
    print("\nper real hard image:")
    for e, n_ in sorted(zip(errs, names), key=lambda z: -z[0]):
        if n_.startswith("hard"): print(f"  {n_:28s} err={e:5.1f}")

if __name__ == "__main__":
    main()
