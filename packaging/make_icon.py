"""Generate n0face brand assets: app icon (.ico/.png) and horizontal logo.

The mark: a 3/4-view head split down the middle. Left half is a neon-lime low-poly wireframe,
right half dissolves into white glitch blocks streaking off to the right. It's built from
MediaPipe's canonical face mesh (the same generic, anonymous face the app's 3D preview uses).

Outputs (assets/):
  n0face.ico        app icon; detailed art for 64 px and up, a bolder minimal mark below that
  n0face.png        1024 px app icon
  logo.png          horizontal logo: mark + "n0face" wordmark + tagline

    python packaging/make_icon.py
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "assets"

LIME = (124, 255, 46)
WHITE = (236, 242, 238)
BG_TOP, BG_BOTTOM = (10, 14, 11), (3, 5, 4)
FONT_BOLD = "C:/Windows/Fonts/bahnschrift.ttf"
FONT_FALLBACK = "C:/Windows/Fonts/arialbd.ttf"

# Face-mesh landmark groups used to keep the low-poly mesh readable.
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_L = [33, 160, 158, 133, 153, 144]
EYE_R = [362, 385, 387, 263, 373, 380]
FEATURES = [1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 2, 98, 327, 61, 291, 0, 17, 13, 14, 78, 308,
            70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 50, 280, 205, 425, 207, 427, 116, 345,
            123, 352, 147, 376, 187, 411, 214, 434, 135, 364, 169, 394, 32, 262, 18, 200, 199, 175]


# ---- the split face mark --------------------------------------------------------------------


def skull_points() -> np.ndarray:
    """Points on an ellipsoid behind/above the face mesh, giving the mark a full head shape."""
    lat = np.radians(np.linspace(-40, 88, 14))
    lon = np.radians(np.linspace(-180, 180, 36, endpoint=False))
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    unit = np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], axis=-1).reshape(-1, 3)
    center, radii = np.array([0.0, 1.6, -3.4]), np.array([8.4, 10.6, 9.6])
    return center + unit * radii


def head_points(yaw: float = -0.5, pitch: float = -0.04):
    """Project the canonical face plus skull (weak perspective), turned toward the viewer's left.

    Returns (face_pts, tris, facing, skull_pts) with image y pointing down."""
    from n0face import canonical

    face = canonical.load()
    v = face.vertices.astype(np.float64)
    shift = v.mean(axis=0)
    v -= shift
    skull = skull_points() - shift
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    r = (rx @ ry)
    vr = v @ r.T
    sr = skull @ r.T
    pts = np.stack([vr[:, 0], -vr[:, 1]], axis=1)  # image y points down
    skull_pts = np.stack([sr[:, 0], -sr[:, 1]], axis=1)
    tris = face.triangles
    n = np.cross(vr[tris[:, 1]] - vr[tris[:, 0]], vr[tris[:, 2]] - vr[tris[:, 0]])
    flip = 1 if (n[:, 2] > 0).sum() > len(n) / 2 else -1  # orientation of "facing"
    facing = n[:, 2] * flip > 0
    return pts, tris, facing, skull_pts


def low_poly_edges(pts: np.ndarray, keep: list[int], silhouette: np.ndarray, scale: float, offset):
    """Delaunay over a sparse landmark subset -> low-poly edges inside the silhouette."""
    p = pts[keep] * scale + offset
    h, w = silhouette.shape
    sub = cv2.Subdiv2D((0, 0, w, h))
    for x, y in p:
        sub.insert((float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))))
    edges = set()
    for t in sub.getTriangleList():
        tri = t.reshape(3, 2)
        c = tri.mean(axis=0).astype(int)
        if not (0 <= c[0] < w and 0 <= c[1] < h) or silhouette[c[1], c[0]] == 0:
            continue
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges.add(tuple(sorted((tuple(np.round(tri[a]).astype(int)), tuple(np.round(tri[b]).astype(int))))))
    return list(edges), p


def face_mark(size: int, minimal: bool = False, glitch_color=WHITE, seed: int = 7) -> Image.Image:
    """RGBA mark on a transparent background, `size` px square."""
    ss = 3
    n = size * ss
    rng = np.random.default_rng(seed)
    pts, tris, facing, skull = head_points()
    allp = np.vstack([pts, skull])
    span = allp.max(axis=0) - allp.min(axis=0)
    scale = 0.92 * n / span.max()
    offset = np.array([n / 2, n / 2]) - (allp.min(axis=0) + span / 2) * scale
    sp = pts * scale + offset
    sk = skull * scale + offset

    silhouette = np.zeros((n, n), np.uint8)
    for tri in np.round(sp[tris]).astype(np.int32):
        cv2.fillConvexPoly(silhouette, tri, 255)
    # Skull: hull of the skull points that sit behind/above the face, merged with the face.
    hull = cv2.convexHull(np.round(np.vstack([sk, sp[OVAL]])).astype(np.int32))
    skull_mask = np.zeros_like(silhouette)
    cv2.fillConvexPoly(skull_mask, hull, 255)
    face_left = sp[:, 0].min()
    skull_mask[:, : int(face_left + 0.08 * n)] = 0  # keep the face profile (nose, lips) on the left
    silhouette = cv2.bitwise_or(silhouette, skull_mask)
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    xs_all = np.nonzero(silhouette.max(axis=0))[0]
    split = int(xs_all.min() + 0.5 * (xs_all.max() - xs_all.min()))  # vertical split through the head
    ys, xs = np.nonzero(silhouette)
    top, bottom = ys.min(), ys.max()

    base = np.zeros((n, n, 4), np.uint8)
    base[silhouette > 0] = (4, 7, 5, 255)  # dark face fill so the background doesn't show through

    # Left half: low-poly lime wireframe.
    line_w = max(1, int(n * (0.012 if minimal else 0.0042)))
    keep = OVAL[::2] + EYE_L + EYE_R + (FEATURES[::4] if minimal else FEATURES[::2])
    if not minimal:
        interior = rng.choice(np.arange(468), 22, replace=False).tolist()
        keep = keep + interior
    skull_keep = skull[:: (9 if minimal else 6)]
    edges, kp = low_poly_edges(np.vstack([pts, skull_keep]), sorted(set(keep)) + list(range(468, 468 + len(skull_keep))),
                               silhouette, scale, offset)
    wire = np.zeros((n, n, 3), np.uint8)
    for a, b in edges:
        cv2.line(wire, a, b, LIME, line_w, cv2.LINE_AA)
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(wire, contours, -1, LIME, int(line_w * 1.8), cv2.LINE_AA)
    for eye in (EYE_L, EYE_R):
        cv2.polylines(wire, [np.round(sp[eye]).astype(np.int32)], True, LIME, int(line_w * 1.8), cv2.LINE_AA)
    if not minimal:
        for x, y in np.round(kp).astype(int)[:: 2]:
            cv2.circle(wire, (int(x), int(y)), line_w + 1, LIME, -1, cv2.LINE_AA)
    wire[:, split:] = 0

    # Right half: glitch blocks within the silhouette, streaking off to the right.
    glitch = np.zeros((n, n), np.float32)
    row = top
    while row < bottom:
        hgt = int(rng.integers(max(2, n // (60 if minimal else 140)), n // (22 if minimal else 45)))
        mid = min(row + hgt // 2, n - 1)
        cols = np.nonzero(silhouette[mid])[0]
        if len(cols):
            right = cols.max()
            x = split + 1
            while x < right:
                w = int(rng.integers(n // 60, n // (10 if minimal else 12)))
                if rng.random() < (0.85 if minimal else 0.74):
                    glitch[row : row + hgt, x : min(x + w, right)] = rng.uniform(0.88, 1.0)
                x += w + int(rng.integers(0, n // 70))
            if rng.random() < 0.45:  # streak past the edge
                length = int(rng.integers(n // 30, n // (5 if minimal else 6)))
                thick = max(2, hgt // int(rng.integers(1, 4)))
                glitch[row : row + thick, right - n // 40 : min(right + length, n - 1)] = rng.uniform(0.6, 1.0)
        row += hgt + int(rng.integers(0, max(2, n // 160)))
    if not minimal:  # pixel grain inside the blocks
        glitch *= rng.uniform(0.82, 1.0, glitch.shape).astype(np.float32)
    glitch[:, : split + 1] = 0

    color = np.array(glitch_color, np.float32)
    glitch_rgb = (glitch[..., None] * color).astype(np.uint8)
    split_line = np.zeros((n, n, 3), np.uint8)
    cv2.line(split_line, (split, top), (split, bottom), WHITE, max(1, line_w // 2), cv2.LINE_AA)

    light = cv2.add(cv2.add(wire, glitch_rgb), split_line)
    glow = cv2.GaussianBlur(light, (0, 0), n * 0.012)
    light_a = light.max(axis=2)
    glow_a = (glow.max(axis=2) * 0.7).astype(np.uint8)

    img = Image.fromarray(base, "RGBA")
    glow_img = Image.fromarray(np.dstack([glow, glow_a]), "RGBA")
    light_img = Image.fromarray(np.dstack([light, light_a]), "RGBA")
    img = Image.alpha_composite(Image.alpha_composite(Image.new("RGBA", (n, n)), glow_img), img)
    img = Image.alpha_composite(img, light_img)
    return img.resize((size, size), Image.LANCZOS)


# ---- app icon ---------------------------------------------------------------------------------


def app_icon(size: int, minimal: bool) -> Image.Image:
    ss = 3
    n = size * ss
    grad = np.linspace(0, 1, n)[:, None, None]
    tile = (np.array(BG_TOP) * (1 - grad) + np.array(BG_BOTTOM) * grad).astype(np.uint8)
    img = Image.fromarray(np.ascontiguousarray(np.broadcast_to(tile, (n, n, 3)))).convert("RGBA")
    radius = int(n * 0.2)
    inset = int(n * 0.035)
    stroke = max(2, int(n * (0.03 if minimal else 0.014)))

    # Faint crosshair ticks (detailed sizes only).
    if not minimal:
        d = ImageDraw.Draw(img)
        tick = (124, 255, 46, 70)
        for fx, fy in ((0.18, 0.5), (0.82, 0.5), (0.5, 0.13), (0.5, 0.87), (0.25, 0.25), (0.75, 0.75)):
            cx, cy, s = fx * n, fy * n, n * 0.018
            d.line([(cx - s, cy), (cx + s, cy)], fill=tick, width=max(1, n // 400))
            d.line([(cx, cy - s), (cx, cy + s)], fill=tick, width=max(1, n // 400))

    mark = face_mark(int(n * (0.8 if minimal else 0.74)), minimal=minimal, glitch_color=LIME if minimal else WHITE)
    img.alpha_composite(mark, ((n - mark.width) // 2, (n - mark.height) // 2))

    # Glowing lime border.
    border = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ImageDraw.Draw(border).rounded_rectangle(
        [inset, inset, n - 1 - inset, n - 1 - inset], radius=radius - inset, outline=LIME + (255,), width=stroke
    )
    glow = border.filter(ImageFilter.GaussianBlur(stroke * 1.6))
    img = Image.alpha_composite(Image.alpha_composite(img, glow), border)

    shape = Image.new("L", (n, n), 0)
    ImageDraw.Draw(shape).rounded_rectangle([0, 0, n - 1, n - 1], radius=radius, fill=255)
    img.putalpha(shape)
    return img.resize((size, size), Image.LANCZOS)


# ---- horizontal logo --------------------------------------------------------------------------


def _font(size: int, weight: str = "Bold") -> ImageFont.FreeTypeFont:
    try:
        f = ImageFont.truetype(FONT_BOLD, size)
        f.set_variation_by_name(weight)
        return f
    except (OSError, ValueError):
        return ImageFont.truetype(FONT_FALLBACK, size)


def wordmark(height: int, seed: int = 3) -> Image.Image:
    """'n0face' with 'n0' in lime, a slashed 0, and horizontal glitch slicing."""
    rng = np.random.default_rng(seed)
    font = _font(int(height * 1.25))
    pad = int(height * 0.9)
    canvas = Image.new("RGBA", (int(height * 6.2) + 2 * pad, int(height * 1.6)), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    x = pad
    baseline_top = int(height * 0.05)
    stem = int(height * 0.2)
    for ch, color in (("n", LIME), ("0", LIME), ("f", WHITE), ("a", WHITE), ("c", WHITE), ("e", WHITE)):
        if ch == "0":
            # Slashed zero drawn by hand to match the stroke weight of the font.
            bbox = d.textbbox((x, baseline_top), "o", font=font)
            w0 = bbox[2] - bbox[0]
            top = d.textbbox((x, baseline_top), "n", font=font)[1] - int(height * 0.32)
            box = [x, top, x + w0, bbox[3]]
            d.rounded_rectangle(box, radius=int(w0 * 0.42), outline=color + (255,), width=int(stem * 0.8))
            d.line([(x + w0 * 0.72, top + stem * 1.1), (x + w0 * 0.28, bbox[3] - stem * 1.1)], fill=color + (255,), width=int(stem * 0.8))
            x += w0 + int(height * 0.12)
        else:
            d.text((x, baseline_top), ch, font=font, fill=color + (255,))
            x += int(d.textlength(ch, font=font)) + int(height * (0.12 if ch == "n" else 0.04))
    text_right = x

    arr = np.array(canvas)
    ys = np.nonzero(arr[..., 3].max(axis=1))[0]
    y0, y1 = ys.min(), ys.max()
    # Slice into horizontal bands and shear some of them sideways.
    out = np.zeros_like(arr)
    row = y0
    while row <= y1:
        hgt = int(rng.integers(max(2, height // 30), height // 7))
        shift = int(rng.normal(0, height * 0.025)) if rng.random() < 0.35 else 0
        out[row : row + hgt] = np.roll(arr[row : row + hgt], shift, axis=1)
        row += hgt
    # Speed lines off the left (lime) and right (white) ends.
    for _ in range(9):
        y = int(rng.integers(y0, y1))
        t = max(2, int(height * rng.uniform(0.02, 0.05)))
        if rng.random() < 0.5:
            x0 = pad - int(rng.integers(height // 4, int(height * 0.8)))
            out[y : y + t, x0 : pad + int(height * 0.15)] = LIME + (230,)
        else:
            x1 = text_right + int(rng.integers(height // 5, int(height * 0.7)))
            out[y : y + t, text_right - int(height * 0.2) : x1] = WHITE + (220,)
    img = Image.fromarray(out)
    glow = img.filter(ImageFilter.GaussianBlur(height * 0.05))
    glow.putalpha(glow.getchannel("A").point(lambda a: int(a * 0.5)))
    return Image.alpha_composite(glow, img).crop(img.getbbox())


def tagline(height: int) -> Image.Image:
    font = _font(height, "Light")
    parts = [("//", LIME), ("   L O C A L    F A C E    P R I V A C Y   ", (200, 208, 204)), ("//", LIME)]
    w = sum(int(ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(t, font=font)) for t, _ in parts)
    img = Image.new("RGBA", (w + 10, int(height * 1.4)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = 0
    for text, color in parts:
        d.text((x, 0), text, font=font, fill=color + (255,))
        x += int(d.textlength(text, font=font))
    return img.crop(img.getbbox())


def logo() -> Image.Image:
    mark = face_mark(420)
    word = wordmark(150)
    tag = tagline(40)
    gap = 30
    w = mark.width + gap + max(word.width, tag.width) + 60
    h = 460
    img = Image.new("RGBA", (w, h), BG_BOTTOM + (255,))
    img.alpha_composite(mark, (20, (h - mark.height) // 2))
    tx = 20 + mark.width + gap
    img.alpha_composite(word, (tx, h // 2 - word.height + 10))
    img.alpha_composite(tag, (tx + (word.width - tag.width) // 2, h // 2 + 50))
    return img


def main() -> None:
    OUT.mkdir(exist_ok=True)
    detailed = app_icon(1024, minimal=False)
    detailed.save(OUT / "n0face.png")
    small = {s: app_icon(s, minimal=True) for s in (16, 20, 24, 32, 40, 48)}
    large = {s: app_icon(s, minimal=False) for s in (64, 128, 256)}
    frames = {**small, **large}
    sizes = sorted(frames)
    frames[256].save(OUT / "n0face.ico", sizes=[(s, s) for s in sizes], append_images=[frames[s] for s in sizes if s != 256])
    logo().save(OUT / "logo.png")
    print(f"wrote n0face.png, n0face.ico ({', '.join(map(str, sizes))} px), logo.png in {OUT}")


if __name__ == "__main__":
    main()
