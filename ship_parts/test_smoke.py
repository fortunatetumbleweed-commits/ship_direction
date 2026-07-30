"""
Smoke test: the extracted standalone estimator must reproduce the in-repo results
(clean 0.7 deg, occluded 9/9 within 20 deg, mean 6.1 deg) on the 21 labelled frames.

The dataset is needed only HERE, to verify the extraction -- never at inference time.

Run:  python ship_parts/test_smoke.py
"""
import os, re, glob, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ship_parts import ShipHeading

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "heading_v9d_raw_images")
EXPECT = {"synth_mean": 0.7, "hard_mean": 6.1, "hard_within20": 9}

def circ_err(a, b):
    return abs((a - b + 180) % 360 - 180)

def main():
    est = ShipHeading()
    print(f"model meta: v{est.meta.get('version')}  val={est.meta.get('val')}\n")

    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "*truth*.png"))):
        img = np.asarray(Image.open(f).convert("RGB"))
        truth = int(re.search(r"truth(\d+)", f).group(1))
        r = est(img)
        rows.append((os.path.basename(f), truth, r.heading, circ_err(r.heading, truth), r.parts_seen))

    hard = np.array([e for n, _, _, e, _ in rows if n.startswith("hard")])
    synth = np.array([e for n, _, _, e, _ in rows if n.startswith("synth")])

    for n, t, p, e, seen in sorted(rows, key=lambda z: -z[3]):
        if n.startswith("hard"):
            print(f"  {n:28s} truth={t:3d} pred={p:6.1f} err={e:5.1f}  parts={','.join(seen) or '-'}")
    got = {"synth_mean": round(float(synth.mean()), 1),
           "hard_mean": round(float(hard.mean()), 1),
           "hard_within20": int((hard <= 20).sum())}
    print(f"\ngot      {got}\nexpected {EXPECT}")

    bad = [k for k, v in EXPECT.items() if abs(got[k] - v) > 0.15]
    if bad:
        print(f"\nFAIL: extraction drifted on {bad}"); return 1
    print("\nPASS: standalone module reproduces the in-repo results")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
