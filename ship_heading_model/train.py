"""
Train a small CNN to estimate ship heading (0-359, 0=up, cw) from an 80x80 image,
robust to occlusion. Heading is circular, so the net regresses (sin, cos).

Data is synthetic (see datagen.py) so it can use scene context the template matcher
can't. Validation is on the 21 REAL labelled images (their truthNNN), reported each
epoch as the honest generalization signal.

Run:  python train.py [--n-train K] [--epochs N] [--version V]
Saves a self-describing checkpoint to model_v{VERSION}.pt (state_dict + a `meta`
block: version, timestamp, training config, validation metrics, and a hash of the
training code). Bump MODEL_VERSION whenever you change the training setup.
"""
import os, re, glob, time, argparse, random, hashlib, datetime
from copy import deepcopy
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import datagen as dg

HERE = os.path.dirname(os.path.abspath(__file__))
W = 80
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_VERSION = 1   # bump when the training setup (datagen / arch / recipe) changes

def code_hash():
    """Short hash of the training pipeline, to tell whether two models share a recipe."""
    h = hashlib.sha256()
    for fn in ("datagen.py", "train.py"):
        with open(os.path.join(HERE, fn), "rb") as f: h.update(f.read())
    return h.hexdigest()[:12]

def to_tensor(imgs_u8):
    x = torch.from_numpy(imgs_u8.astype(np.float32) / 255.0 - 0.5)
    return x.permute(0, 3, 1, 2).contiguous()   # N,3,80,80

def deg_to_vec(deg):
    r = np.radians(deg)
    return np.stack([np.sin(r), np.cos(r)], -1).astype(np.float32)

def vec_to_deg(v):
    return np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360

def circ_err(a, b):
    return np.abs((a - b + 180) % 360 - 180)

class HeadingNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o): return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 24), block(24, 48), block(48, 96), block(96, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        v = self.head(self.features(x))
        return v / v.norm(dim=1, keepdim=True).clamp_min(1e-6)

def load_real(raw_dir=dg.RAW_DIR):
    imgs, degs, names = [], [], []
    for f in sorted(glob.glob(os.path.join(raw_dir, "*truth*.png"))):
        imgs.append(np.asarray(Image.open(f).convert("RGB")))
        degs.append(int(re.search(r"truth(\d+)", f).group(1)))
        names.append(os.path.basename(f))
    return to_tensor(np.stack(imgs)), np.array(degs), names

@torch.no_grad()
def evaluate(model, x, degs, names):
    model.eval()
    pred = vec_to_deg(model(x.to(DEVICE)).cpu().numpy())
    err = circ_err(pred, degs)
    hard = np.array([e for e, n in zip(err, names) if n.startswith("hard")])
    synth = np.array([e for e, n in zip(err, names) if n.startswith("synth")])
    return err, hard, synth

def gen_dataset(sprite, bgs, n):
    X = np.empty((n, W, W, 3), np.uint8); Y = np.empty((n,), np.float32)
    for i in range(n):
        im, h = dg.sample(sprite, bgs); X[i] = im; Y[i] = h
    return X, Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", default=str(MODEL_VERSION), help="version tag for the saved checkpoint")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"device={DEVICE}")
    sprite = dg.build_ship_sprite(); bgs = dg.harvest_backgrounds()
    print(f"generating {args.n_train} synthetic samples ...")
    t0 = time.time(); Xtr, Ytr = gen_dataset(sprite, bgs, args.n_train)
    print(f"  done in {time.time()-t0:.0f}s")
    Xtr_t = to_tensor(Xtr); Ytr_t = torch.from_numpy(deg_to_vec(Ytr))
    xr, degs, names = load_real()

    model = HeadingNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.MSELoss()
    n = Xtr_t.shape[0]; best = 1e9; best_state = None
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i+args.batch]
            xb = Xtr_t[idx].to(DEVICE); yb = Ytr_t[idx].to(DEVICE)
            opt.zero_grad(); pred = model(xb); loss = lossf(pred, yb)
            loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        err, hard, synth = evaluate(model, xr, degs, names)
        tag = ""
        if hard.mean() < best:
            best = hard.mean(); best_state = deepcopy(model.state_dict()); tag = "  *best"
        print(f"ep{ep:02d} loss={tot/n:.4f}  synth_mean={synth.mean():5.1f}  "
              f"hard_mean={hard.mean():5.1f} hard_median={np.median(hard):5.1f} "
              f"hard_within20={(hard<=20).sum()}/{len(hard)}{tag}")

    # ---- save a self-describing, versioned checkpoint of the best model ----
    model.load_state_dict(best_state)
    err, hard, synth = evaluate(model, xr, degs, names)
    meta = {
        "version": args.version,
        "arch": "HeadingNet",
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_hash": code_hash(),
        "args": {k: getattr(args, k) for k in ("n_train", "epochs", "batch", "seed")},
        "val": {"hard_mean": round(float(hard.mean()), 2),
                "hard_median": round(float(np.median(hard)), 2),
                "hard_within20": int((hard <= 20).sum()), "n_hard": int(len(hard)),
                "synth_mean": round(float(synth.mean()), 2)},
    }
    path = os.path.join(HERE, f"model_v{args.version}.pt")
    torch.save({"state_dict": best_state, "meta": meta}, path)
    print(f"\nsaved {os.path.basename(path)}")
    for k, v in meta.items(): print(f"  {k}: {v}")
    print("\nfinal (best model) per real hard image:")
    for e, n_ in sorted(zip(err, names), key=lambda z: -z[0]):
        if n_.startswith("hard"):
            print(f"  {n_:28s} err={e:5.1f}")

if __name__ == "__main__":
    main()
