"""
Synthetic training-data generator for ship heading estimation under occlusion.

Occlusion is modeled the way the real game works: the ship sails UNDER whole,
opaque overlays drawn on the map. Real occluders (harvested / matched from the
hard_* images):
  - NPC portrait icon  (portrait.png, extracted from a hard_* image)
  - village-name text  ("<Name> Village", bold white with dark outline)
  - minimap icons      (house marker, diamond marker)
  - terrain "cover"    (ship passing behind land) -- secondary
Backgrounds are purity-filtered terrain tiles (no portrait/text fragments) and
mirror-tiled to avoid the fragment-repeat artifact.

Every sample keeps a minimum visible hull, so we never label a fully-hidden ship.
Point RAW_DIR at game exports (or drop more portrait_*.png assets) to shrink the
domain gap further.
"""
import glob, os, re, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "..", "heading_v9d_raw_images")
W = 80
CENTER = (40.0, 40.0)
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
NAMES = ["Nub", "Bari", "Nuiba", "Anga", "Toro", "Kai", "Mira", "Sabo",
         "Loka", "Zuri", "Vega", "Pako", "Rin", "Odo", "Suri", "Tavo"]
SUFFIX = ["Village", "Village", "Village", "Port", "Bay", "Town"]

def _rgb(f): return np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
def _ship_mask(a, loose=True):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if loose: return (g > 90) & (g > r + 30) & (g > b + 30)
    return (g > 110) & (g - r > 60) & (g - b > 50)

# ---------- assets ----------
def build_ship_sprite(raw_dir=RAW_DIR):
    """Clean colored ship RGBA at heading 0 (median of de-rotated synths, L/R folded)."""
    rgbs, alphas = [], []
    for f in sorted(glob.glob(os.path.join(raw_dir, "synth_*truth*.png"))):
        t = int(re.search(r"truth(\d+)", f).group(1))
        a = _rgb(f); m = _ship_mask(a).astype(np.float32)
        rgba = np.dstack([a, m * 255]).astype(np.uint8)
        der = Image.fromarray(rgba, "RGBA").rotate(+t, resample=Image.BICUBIC, center=CENTER)
        d = np.asarray(der).astype(np.float32)
        rgbs.append(d[..., :3]); alphas.append(d[..., 3])
    A = np.stack(alphas); mA = np.median(A, 0)
    wsum = (A > 60).astype(np.float32)
    mC = (np.stack(rgbs) * (A > 60)[..., None]).sum(0) / np.maximum(wsum.sum(0), 1)[..., None]
    sprite = np.dstack([mC, mA]); sprite = np.maximum(sprite, np.fliplr(sprite))
    return sprite.clip(0, 255).astype(np.uint8)

def load_portraits():
    ps = [Image.open(f).convert("RGBA") for f in sorted(glob.glob(os.path.join(HERE, "portrait*.png")))
          if "preview" not in f]
    return ps

def _font(size):
    try: return ImageFont.truetype(FONT_PATH, size)
    except Exception: return ImageFont.load_default()

def harvest_backgrounds(raw_dir=RAW_DIR, tile=22):
    """Pure-terrain tiles: reject any tile containing ship-green, portrait-yellow, or near-white."""
    tiles = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "hard_*truth*.png"))):
        a = _rgb(f); r, g, b = a[..., 0], a[..., 1], a[..., 2]
        shipg = (g > 110) & (g - r > 60) & (g - b > 50)
        yellow = (r > 150) & (g > 110) & (r > b + 45) & (g > b + 25)
        white = (r > 200) & (g > 200) & (b > 200)
        bad = shipg | yellow | white
        for _ in range(250):
            y, x = random.randint(0, W - tile), random.randint(0, W - tile)
            if bad[y:y+tile, x:x+tile].mean() < 0.02:
                tiles.append(a[y:y+tile, x:x+tile].copy())
    return tiles

# ---------- rendering primitives ----------
def render_ship(sprite, heading):
    im = Image.fromarray(sprite, "RGBA").rotate(-heading % 360, resample=Image.BICUBIC, center=CENTER)
    return np.asarray(im).astype(np.float32)

def _bg_canvas(bg_tiles):
    """One terrain tile mirror-tiled to 80x80 (coherent, seamless)."""
    t = random.choice(bg_tiles)
    s = random.randint(38, 48)
    base = Image.fromarray(t.clip(0, 255).astype(np.uint8)).resize((s, s), Image.BILINEAR)
    canvas = Image.new("RGB", (s * 2, s * 2))
    canvas.paste(base, (0, 0))
    canvas.paste(base.transpose(Image.FLIP_LEFT_RIGHT), (s, 0))
    canvas.paste(base.transpose(Image.FLIP_TOP_BOTTOM), (0, s))
    canvas.paste(base.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (s, s))
    ox, oy = random.randint(0, max(0, 2*s - W)), random.randint(0, max(0, 2*s - W))
    out = np.asarray(canvas.crop((ox, oy, ox + W, oy + W)).resize((W, W))).astype(np.float32)
    return (out + np.random.randn(W, W, 3) * random.uniform(0, 5)).clip(0, 255)

def _stamp(sprite_pil, allow_flip=True):
    """Place an RGBA sprite at a random transform on an 80x80 canvas; return (rgb, alpha01)."""
    sp = sprite_pil
    sc = random.uniform(0.7, 1.3)
    sp = sp.resize((max(1, int(sp.width * sc)), max(1, int(sp.height * sc))), Image.BILINEAR)
    if allow_flip and random.random() < 0.5: sp = sp.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5: sp = sp.rotate(random.uniform(-15, 15), resample=Image.BICUBIC, expand=True)
    lay = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    cx = random.randint(15, 65); cy = random.randint(15, 65)
    lay.paste(sp, (int(cx - sp.width / 2), int(cy - sp.height / 2)), sp)
    arr = np.asarray(lay).astype(np.float32)
    return arr[..., :3], arr[..., 3] / 255.0

def _text_sprite():
    name = random.choice(NAMES) + " " + random.choice(SUFFIX)
    fnt = _font(random.randint(11, 18))
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = dummy.textbbox((0, 0), name, font=fnt, stroke_width=2)
    im = Image.new("RGBA", (r - l + 6, b - t + 6), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((3 - l, 3 - t), name, font=fnt, fill=(245, 245, 245, 255),
           stroke_width=2, stroke_fill=(45, 45, 45, 255))
    return im

def _house_sprite():
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(2, 7), (8, 1), (14, 7)], fill=(235, 235, 235), outline=(40, 40, 40))
    d.rectangle([4, 7, 12, 14], fill=(235, 235, 235), outline=(40, 40, 40))
    return im

def _diamond_sprite():
    im = Image.new("RGBA", (14, 14), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(7, 0), (14, 7), (7, 14), (0, 7)], fill=(240, 240, 240), outline=(60, 60, 60))
    return im

def _cover_layer(bg_tiles):
    cover = _bg_canvas(bg_tiles)
    mask = Image.new("L", (W, W), 0); dd = ImageDraw.Draw(mask)
    ang = math.radians(random.uniform(0, 360))
    cx, cy = random.uniform(25, 55), random.uniform(25, 55)
    dx, dy = math.cos(ang), math.sin(ang); f = 200
    bx, by = cx + dy * f, cy - dx * f
    dd.polygon([(bx + dx * f, by + dy * f), (bx - dx * f, by - dy * f),
                (bx - dx * f - dy * 300, by - dy * f + dx * 300),
                (bx + dx * f - dy * 300, by + dy * f + dx * 300)], fill=255)
    return cover, np.asarray(mask).astype(np.float32) / 255.0

# ---------- sample ----------
def _occluder_layers(bg_tiles, portraits):
    layers = []
    if portraits and random.random() < 0.55: layers.append(_stamp(random.choice(portraits)))
    if random.random() < 0.5:                layers.append(_stamp(_text_sprite(), allow_flip=False))
    for _ in range(random.randint(0, 2)):
        if random.random() < 0.5:            layers.append(_stamp(_house_sprite()))
        else:                                layers.append(_stamp(_diamond_sprite()))
    if random.random() < 0.25:               layers.append(_cover_layer(bg_tiles))
    return layers

def sample(sprite, bg_tiles, portraits=None, occ_prob=0.9, min_visible_px=18, tries=6):
    """Return (uint8 HxWx3 image, heading_degrees), guaranteeing a minimum visible hull."""
    if portraits is None: portraits = load_portraits()
    heading = random.uniform(0, 360)
    bg = _bg_canvas(bg_tiles)
    ship = render_ship(sprite, heading)
    jx, jy = random.randint(-3, 3), random.randint(-3, 3)
    ship = np.roll(np.roll(ship, jy, 0), jx, 1)
    ship_a = ship[..., 3] > 60; n_ship = int(ship_a.sum())
    target = min(min_visible_px, max(1, n_ship // 3))
    ship_rgb, ship_al = ship[..., :3], (ship[..., 3:4] / 255.0)

    best_img, best_vis = None, -1
    for _ in range(tries):
        img = bg.copy(); img[:] = ship_rgb * ship_al + img * (1 - ship_al)
        occ = np.zeros((W, W), np.float32)
        if random.random() < occ_prob:
            for rgb, m in _occluder_layers(bg_tiles, portraits):
                m3 = m[..., None]; img[:] = rgb * m3 + img * (1 - m3)
                occ = np.maximum(occ, m)
        visible = int((ship_a & (occ < 0.5)).sum())
        if visible > best_vis: best_vis, best_img = visible, img
        if visible >= target: break
    img = best_img + np.random.randn(W, W, 3) * random.uniform(0, 4)
    return img.clip(0, 255).astype(np.uint8), heading

if __name__ == "__main__":
    random.seed(0); np.random.seed(0)
    sprite = build_ship_sprite(); Image.fromarray(sprite, "RGBA").save(os.path.join(HERE, "ship0.png"))
    bgs = harvest_backgrounds(); ports = load_portraits()
    print(f"bg tiles: {len(bgs)}  portraits: {len(ports)}")
    n, cols, pad = 48, 8, 4; rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (W + pad) + pad, rows * (W + pad) + pad), (30, 30, 30))
    for i in range(n):
        im, h = sample(sprite, bgs, ports)
        sheet.paste(Image.fromarray(im), (pad + (i % cols) * (W + pad), pad + (i // cols) * (W + pad)))
    sheet.save(os.path.join(HERE, "sample_sheet.png"))
    print("wrote ship0.png, sample_sheet.png")
