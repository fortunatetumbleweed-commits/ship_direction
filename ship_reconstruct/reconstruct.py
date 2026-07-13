#!/usr/bin/env python3
"""
Ship heading estimation + de-occlusion by template matching.

Given an 80x80 image of a green ship icon (possibly occluded by map/UI overlays)
at an UNKNOWN heading, this:
  1. estimates the heading by matching a clean ship template through all 360 deg,
  2. registers (translates) the template onto the visible ship pixels,
  3. writes a clean reconstructed ship sprite and a de-occluded overlay,
  4. reports a confidence so you know when a frame is too occluded to trust.

No training, no GPU. Only needs: numpy, pillow.

Usage
-----
  # build the reusable template once from the clean synth_* images:
  python reconstruct.py build-template --synth-dir /path/to/raw_images --out template.png

  # reconstruct one or many images (heading estimated automatically):
  python reconstruct.py run img1.png img2.png --template template.png --out-dir out/

  # force a heading (for frames too occluded to trust the estimate):
  python reconstruct.py run img.png --template template.png --angle 180 --out-dir out/

  # measure accuracy against ground-truth 'truthNNN' in filenames:
  python reconstruct.py validate *.png --template template.png
"""
import argparse, glob, os, re, sys, math
import numpy as np
from PIL import Image

W = 80
CENTER = (40.0, 40.0)

# ---------- segmentation ----------
def rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)

def ship_mask(a, loose=False):
    """Boolean mask of bright saturated ship-green (rejects duller grass-green)."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if loose:
        return (g > 90) & (g > r + 30) & (g > b + 30)
    return (g > 110) & (g - r > 60) & (g - b > 50)

# ---------- template ----------
def build_template(synth_dir):
    """Canonical ship alpha at heading 0, from de-rotated clean synth_* images."""
    files = sorted(glob.glob(os.path.join(synth_dir, "synth_*truth*.png")))
    if not files:
        sys.exit(f"no synth_*truth*.png found in {synth_dir}")
    stack = []
    for f in files:
        t = int(re.search(r"truth(\d+)", f).group(1))
        a = rgb(f)
        m = ship_mask(a, loose=True).astype(np.float32)
        rgba = np.dstack([a, m * 255]).astype(np.uint8)
        # de-rotate a heading-t ship back to heading 0  (inverse of render below)
        der = Image.fromarray(rgba, "RGBA").rotate(+t, resample=Image.BICUBIC, center=CENTER)
        stack.append(np.asarray(der)[..., 3].astype(np.float32))
    alpha = np.median(np.stack(stack), 0)
    # ship is left/right symmetric about its fore-aft axis (vertical at heading 0),
    # so fold the template with its mirror to remove sampling noise.
    alpha = np.maximum(alpha, np.fliplr(alpha))
    alpha = (alpha > 60).astype(np.uint8) * 255
    return Image.fromarray(alpha)  # 'L' image, heading 0

def render(template, angle):
    """Binary template footprint rotated to `angle` (0=up, clockwise)."""
    r = template.rotate(-angle % 360, resample=Image.BICUBIC, center=CENTER)
    return (np.asarray(r).astype(np.float32) / 255.0 > 0.5).astype(np.float32)

def render_sprite(template, src_rgb_for_color, angle, shift):
    """Clean green RGBA ship sprite at (angle, shift), colored like the ship."""
    foot = render(template, angle)
    dy, dx = shift
    foot = np.roll(np.roll(foot, dy, 0), dx, 1)
    color = np.array([95, 190, 52], np.float32)  # canonical ship green
    rgba = np.zeros((W, W, 4), np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = foot * 255
    return Image.fromarray(rgba.clip(0, 255).astype(np.uint8), "RGBA")

# ---------- matching ----------
def _best_shift_score(S, T, R):
    """Over integer shifts in +-R, return (score, dy, dx).
    score = coverage(visible explained) * fill(fragment fills template) -> robust to occlusion."""
    nS = S.sum()
    if nS == 0:
        return 0.0, 0, 0
    best = (-1.0, 0, 0)
    for dy in range(-R, R + 1):
        Ty = np.roll(T, dy, 0)
        for dx in range(-R, R + 1):
            Ts = np.roll(Ty, dx, 1)
            inter = float((S * Ts).sum())
            if inter == 0:
                continue
            cov = inter / nS               # how much of the visible ship the template covers
            fill = inter / float(Ts.sum()) # how much of the template the fragment fills
            sc = cov * (fill ** 0.3)
            if sc > best[0]:
                best = (sc, dy, dx)
    return best

def estimate(template, img_rgb, coarse=3, R=10):
    S = ship_mask(img_rgb).astype(np.float32)
    nvis = int(S.sum())
    scores = {}
    best = (-1.0, 0, 0, 0)  # score, angle, dy, dx
    for a in range(0, 360, coarse):
        sc, dy, dx = _best_shift_score(S, render(template, a), R)
        scores[a] = sc
        if sc > best[0]:
            best = (sc, a, dy, dx)
    for a in range(best[1] - coarse, best[1] + coarse + 1):  # refine to 1 deg
        sc, dy, dx = _best_shift_score(S, render(template, a % 360), R)
        if sc > best[0]:
            best = (sc, a % 360, dy, dx)
    score, angle, dy, dx = best
    def axis_diff(a):   # 0..90, difference ignoring bow/stern direction
        return abs((a - angle + 90) % 180 - 90)
    # competitor that points the *opposite* way but same axis -> bow/stern flip risk
    flip = max((s for a, s in scores.items() if circ_err(a, angle) >= 150), default=0.0)
    # competitor on a genuinely *different axis* -> axis ambiguity
    cross = max((s for a, s in scores.items() if axis_diff(a) >= 40), default=0.0)
    return {"angle": angle, "axis": angle % 180, "shift": (dy, dx), "score": score,
            "visible_px": nvis, "flip_score": flip, "cross_score": cross}

def _margin(best, comp, th):
    return min(max((best - comp) / max(best, 1e-6) / th, 0.0), 1.0)

def axis_confidence(res):
    """0..1 confidence in the AXIS (0-180), independent of bow/stern direction."""
    vis_term = min(res["visible_px"] / 90.0, 1.0)    # axis needs less signal than direction
    fit_term = min(res["score"] / 0.6, 1.0)
    return round(float(vis_term * fit_term * _margin(res["score"], res["cross_score"], 0.15)), 2)

def dir_confidence(res):
    """0..1 confidence in the bow/stern DIRECTION (the front/back call)."""
    return round(float(_margin(res["score"], res["flip_score"], 0.15)), 2)

def confidence(res):
    """Overall heading confidence = need BOTH the axis and the direction."""
    return round(axis_confidence(res) * dir_confidence(res), 2)

# ---------- io ----------
def circ_err(a, b, m=360):
    return abs((a - b + m / 2) % m - m / 2)

def run(args):
    template = Image.open(args.template).convert("L")
    os.makedirs(args.out_dir, exist_ok=True)
    for f in args.images:
        a = rgb(f)
        if args.angle is not None:
            S = ship_mask(a).astype(np.float32)
            _, dy, dx = _best_shift_score(S, render(template, args.angle), 12)
            res = {"angle": args.angle, "axis": args.angle % 180, "shift": (dy, dx),
                   "visible_px": int(S.sum())}
            note = "  (heading forced)"
        else:
            res = estimate(template, a)
            note = _explain(res, args.conf_th)
        sprite = render_sprite(template, a, res["angle"], res["shift"])
        base = os.path.splitext(os.path.basename(f))[0]
        sprite.save(os.path.join(args.out_dir, base + "_ship.png"))
        Image.alpha_composite(Image.open(f).convert("RGBA"), sprite).convert("RGB").save(
            os.path.join(args.out_dir, base + "_deocc.png"))
        print(f"{os.path.basename(f):28s} heading={res['angle']:3d}  "
              f"axis={res['axis']:3d}  visible_px={res['visible_px']:4d}{note}")

def _explain(res, th):
    """Human-readable confidence note distinguishing axis vs bow/stern certainty."""
    ac, dc = axis_confidence(res), dir_confidence(res)
    if ac < th:
        return f"  <-- LOW CONFIDENCE: ship too occluded (axis_conf={ac})"
    if dc < th:
        flip = (res["angle"] + 180) % 360
        return (f"  <-- AXIS OK (conf={ac}) but BOW/STERN uncertain "
                f"(could be {flip}); set --angle if known")
    return f"  (confident: axis={ac}, direction={dc})"

def validate(args):
    template = Image.open(args.template).convert("L")
    rows = []
    for f in sorted(args.images):
        m = re.search(r"truth(\d+)", f)
        if not m:
            continue
        truth = int(m.group(1))
        res = estimate(template, rgb(f))
        rows.append((os.path.basename(f), truth, res["angle"],
                     circ_err(res["angle"], truth),               # heading error
                     circ_err(res["axis"], truth % 180, 180),     # axis error (mod 180)
                     res["visible_px"], axis_confidence(res), dir_confidence(res)))
    print(f"{'file':30s} {'truth':>5} {'est':>4} {'h_err':>5} {'ax_err':>6} "
          f"{'vis':>4} {'ax_cf':>5} {'dir_cf':>6}")
    for r in rows:
        print(f"{r[0]:30s} {r[1]:5d} {r[2]:4d} {r[3]:5.0f} {r[4]:6.0f} "
              f"{r[5]:4d} {r[6]:5} {r[7]:6}")
    th = args.conf_th
    axis_ok = np.array([r[4] for r in rows if r[6] >= th], float)     # axis err where axis confident
    head_ok = np.array([r[3] for r in rows if r[6] >= th and r[7] >= th], float)
    print(f"\nN={len(rows)}")
    print(f"AXIS  confident (ax_cf>={th}): N={len(axis_ok)}  "
          f"mean|axis_err|={axis_ok.mean():.1f}  max={axis_ok.max():.0f}")
    print(f"FULL heading confident (both>={th}): N={len(head_ok)}  "
          f"mean|err|={head_ok.mean():.1f}  max={head_ok.max():.0f}")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-template", help="build template.png from clean synth_* images")
    b.add_argument("--synth-dir", required=True)
    b.add_argument("--out", default="template.png")

    r = sub.add_parser("run", help="estimate heading + reconstruct one or more images")
    r.add_argument("images", nargs="+")
    r.add_argument("--template", default="template.png")
    r.add_argument("--out-dir", default="out")
    r.add_argument("--angle", type=int, default=None, help="force heading (deg, 0=up, cw)")
    r.add_argument("--conf-th", type=float, default=0.4)

    v = sub.add_parser("validate", help="report angular error vs truthNNN in filenames")
    v.add_argument("images", nargs="+")
    v.add_argument("--template", default="template.png")
    v.add_argument("--conf-th", type=float, default=0.4)

    args = p.parse_args()
    if args.cmd == "build-template":
        build_template(args.synth_dir).save(args.out)
        print(f"wrote {args.out}")
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "validate":
        validate(args)

if __name__ == "__main__":
    main()
