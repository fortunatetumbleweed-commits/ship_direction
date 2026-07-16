"""
Train the part-segmentation model: a U-Net that labels each pixel as one of
{bg, bow, hull, stern, sail_l, sail_r}. Heading is recovered by matching the predicted
part masks to the rotated canonical part-layout (parts.part_pose) -- the dense, region
version of the keypoint constellation fit. Validation runs the full chain on the real
labelled images.

Run:  python parts_train.py [--n-train K] [--epochs N] [--version V]
Saves part_model_v{VERSION}.pt (self-describing checkpoint).
"""
import os, time, argparse, random, hashlib, datetime
from copy import deepcopy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import parts, keypoints as kp
import datagen as kdg
from train import cbr, to_tensor, load_real, circ_err, DEVICE, W

HERE = os.path.dirname(os.path.abspath(__file__))
NC = 6                                   # bg + 5 parts
MODEL_VERSION = 1

def code_hash():
    h = hashlib.sha256()
    for fn in ("parts.py", "parts_train.py", "datagen.py"):
        with open(os.path.join(HERE, fn), "rb") as f: h.update(f.read())
    return h.hexdigest()[:12]

class SegNet(nn.Module):
    def __init__(self, nc=NC):
        super().__init__()
        self.e1 = nn.Sequential(cbr(3, 32), cbr(32, 32))
        self.e2 = nn.Sequential(cbr(32, 64), cbr(64, 64))
        self.e3 = nn.Sequential(cbr(64, 128), cbr(128, 128))
        self.pool = nn.MaxPool2d(2)
        self.d2 = nn.Sequential(cbr(128 + 64, 64), cbr(64, 64))
        self.d1 = nn.Sequential(cbr(64 + 32, 32), cbr(32, 32))
        self.head = nn.Conv2d(32, nc, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        d = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.d2(torch.cat([d, e2], 1))
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.d1(torch.cat([d, e1], 1))
        return self.head(d)

def sample_seg(sprite, L0, bgs, ports, occ_prob=0.9, min_vis=30, tries=6):
    heading = random.uniform(0, 360); jx, jy = random.randint(-4, 4), random.randint(-4, 4)
    ship = np.roll(np.roll(kdg._render_ship(sprite, heading), jy, 0), jx, 1)
    ship_a = ship[..., 3] > 60; lab = parts.render_labels(L0, heading, (jy, jx))
    a = ship[..., 3:4] / 255.0; best = None
    for _ in range(tries):
        img = kdg.hdg._bg_canvas(bgs); img[:] = ship[..., :3] * a + img * (1 - a)
        occ = np.zeros((W, W), np.float32)
        if random.random() < occ_prob:
            for rgb, m in kdg.hdg._occluder_layers(bgs, ports):
                m3 = m[..., None]; img[:] = rgb * m3 + img * (1 - m3); occ = np.maximum(occ, m)
        target = np.where(ship_a & (occ < 0.5), lab, 0).astype(np.int64)   # visible parts, else bg
        nvis = int((target > 0).sum())
        if best is None or nvis > best[0]: best = (nvis, img.copy(), target)
        if nvis >= min_vis: break
    _, img, target = best
    img = img + np.random.randn(W, W, 3) * random.uniform(0, 4)
    return img.clip(0, 255).astype(np.uint8), target, heading

def gen_dataset(sprite, L0, bgs, ports, n):
    X = np.empty((n, W, W, 3), np.uint8); Y = np.empty((n, W, W), np.int64)
    for i in range(n):
        img, tgt, _ = sample_seg(sprite, L0, bgs, ports); X[i] = img; Y[i] = tgt
    return X, Y

@torch.no_grad()
def validate(model, xr, degs, names, L0):
    model.eval(); errs = []
    for b in range(xr.shape[0]):
        P = torch.softmax(model(xr[b:b+1].to(DEVICE)), 1)[0].cpu().numpy()
        img = ((xr[b].permute(1, 2, 0).cpu().numpy() + 0.5) * 255).clip(0, 255).astype(np.int32)
        _, open_mask = parts.mr.open_mask_from_crop(img)        # scene fallback for degenerate poses
        heading, _ = parts.part_pose(P, L0, open_mask=open_mask)
        errs.append(circ_err(heading, degs[b]))
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

    print(f"device={DEVICE}  classes={NC}")
    sprite, _, bgs, ports = kdg.assets(); L0 = parts.canonical_parts()
    print(f"generating {args.n_train} samples ..."); t0 = time.time()
    X, Y = gen_dataset(sprite, L0, bgs, ports, args.n_train); print(f"  done in {time.time()-t0:.0f}s")
    Xt = to_tensor(X); Yt = torch.from_numpy(Y)
    xr, degs, names = load_real()
    wt = torch.tensor([0.2, 1, 1, 1, 1, 1.], device=DEVICE)         # downweight background

    model = SegNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = Xt.shape[0]; best = 1e9; best_state = None
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i+args.batch]
            xb = Xt[idx].to(DEVICE); yb = Yt[idx].to(DEVICE)
            loss = F.cross_entropy(model(xb), yb, weight=wt)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        errs, hard, synth = validate(model, xr, degs, names, L0)
        tag = ""
        if hard.mean() < best:
            best = hard.mean(); best_state = deepcopy(model.state_dict()); tag = "  *best"
        print(f"ep{ep:02d} loss={tot/n:.4f}  synth_mean={synth.mean():5.1f}  "
              f"hard_mean={hard.mean():5.1f} hard_within20={(hard<=20).sum()}/{len(hard)}{tag}")

    model.load_state_dict(best_state); errs, hard, synth = validate(model, xr, degs, names, L0)
    meta = {"version": args.version, "arch": "SegNet", "parts": parts.PART_NAMES,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"), "code_hash": code_hash(),
            "args": {k_: getattr(args, k_) for k_ in ("n_train", "epochs", "batch", "seed")},
            "inference": "part_pose (dense part-mask fit)",
            "val": {"hard_mean": round(float(hard.mean()), 2), "hard_within20": int((hard <= 20).sum()),
                    "n_hard": int(len(hard)), "synth_mean": round(float(synth.mean()), 2)}}
    path = os.path.join(HERE, f"part_model_v{args.version}.pt")
    torch.save({"state_dict": best_state, "meta": meta}, path)
    print(f"\nsaved {os.path.basename(path)}")
    for k_, v_ in meta.items(): print(f"  {k_}: {v_}")
    print("\nper real hard image:")
    for e, n_ in sorted(zip(errs, names), key=lambda z: -z[0]):
        if n_.startswith("hard"): print(f"  {n_:28s} err={e:5.1f}")

if __name__ == "__main__":
    main()
