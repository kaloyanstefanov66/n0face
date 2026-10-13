"""Export the website's assets from the app itself: mesh data, mask renders, a UI screenshot.

    .venv\\Scripts\\python scripts\\make_site_assets.py

Writes into site/assets/. Everything comes from free (public) code: no PRO plugin is loaded.
The PRO previews in site/assets/extras/ are the EXTRAS tab's pictures rendered in the site's
green by the private repo's tools/make_extras_previews.py; this script doesn't touch them.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "assets"
BRAND = (43, 255, 124)  # BGR of the logo's acid green, #7cff2b


def straight_rgba(premul: np.ndarray) -> np.ndarray:
    a = premul[..., 3:4].astype(np.float32)
    rgb = np.where(a > 0, premul[..., :3].astype(np.float32) * 255 / np.maximum(a, 1), 0)
    return np.dstack([np.clip(rgb, 0, 255), a]).astype(np.uint8)


def face_json() -> None:
    """The canonical (anonymous) face mesh for the hero animation: ~25 KB."""
    from n0face import canonical

    face = canonical.load()
    v = face.vertices.astype(np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    v = (v - (lo + hi) / 2) / 9.0
    tris = face.triangles.copy()
    n = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    if n[:, 2].mean() < 0:
        tris = tris[:, ::-1]
    data = {
        "v": [round(float(x), 3) for x in v.ravel()],
        "t": [int(i) for i in tris.ravel()],
        "e": [int(i) for i in face.edges.ravel()],
        **x_design(v),
    }
    from n0face import wrap

    assert np.array_equal(tris, wrap.mesh().tris), "the X anchors index wrap.mesh() triangles"
    (OUT / "face.json").write_text(json.dumps(data, separators=(",", ":")))


def x_design(v: np.ndarray) -> dict:
    """The X mask's strokes, pinned to mesh triangles exactly as the app pins them, for the three
    stop-motion "boil" frames. Each polyline is flat [triangle, w1, w2, ...] (w0 = 1 - w1 - w2);
    triangle indices refer to wrap.mesh().tris."""
    from n0face import designs, wrap

    m = wrap.mesh()
    boils = []
    for boil in range(3):
        groups = designs._x_shapes(1.0, 1.0, 0.35, 0.0, boil)
        out = []
        for polys in groups:
            lines = []
            for poly in polys:
                anc = designs.anchor(poly)
                flat = []
                for tri, b in zip(anc.tri, anc.bary):
                    flat += [int(tri), round(float(b[1]), 3), round(float(b[2]), 3)]
                lines.append(flat)
            out.append(lines)
        boils.append(out)
    # texels -> mesh units, so stroke widths (given in texels) scale with the face
    uv, tris = m.uv * designs.TEX, m.tris
    e3 = sum(np.linalg.norm(v[tris[:, i]] - v[tris[:, (i + 1) % 3]], axis=1).sum() for i in range(3))
    e2 = sum(np.linalg.norm(uv[tris[:, i]] - uv[tris[:, (i + 1) % 3]], axis=1).sum() for i in range(3))
    return {"x": boils, "k": round(float(e3 / e2), 6)}


def mask_renders() -> None:
    from n0face import wrap
    from n0face.app.preview3d import MaskScene, Pose
    from n0face.theme import Theme

    theme = Theme(primary=BRAND, accent=(235, 235, 235))
    scene = MaskScene()
    size = 900
    for name, effect, backing, yaw in (("mask-mesh", "mesh", "solid", 0.32), ("mask-x", "x", "solid", -0.32)):
        wrap.config.enabled, wrap.config.backing = True, backing
        frame = scene.render(size, size, effect, "mask", theme, 1.0, Pose(yaw, 0.06), px=2.0)
        canvas = np.zeros((size, size, 4), np.uint8)
        canvas[frame.y:frame.y + frame.image.shape[0], frame.x:frame.x + frame.image.shape[1]] = frame.image
        ys, xs = np.nonzero(canvas[..., 3] > 8)
        cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
        half = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.56)
        crop = np.zeros((2 * half, 2 * half, 4), np.uint8)
        y0, x0 = cy - half, cx - half
        sy, sx = max(y0, 0), max(x0, 0)
        part = canvas[sy:cy + half, sx:cx + half]
        crop[sy - y0:sy - y0 + part.shape[0], sx - x0:sx - x0 + part.shape[1]] = part
        tile = cv2.resize(straight_rgba(crop), (720, 720), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"{name}.webp"), tile, [cv2.IMWRITE_WEBP_QUALITY, 92])


def glitch_sprite() -> None:
    """The X mask clean, then three frames of the app's real GLITCH effect, side by side."""
    from n0face import glitch, wrap
    from n0face.app.preview3d import MaskScene, Pose
    from n0face.theme import Theme

    theme = Theme(primary=BRAND, accent=(235, 235, 235))
    wrap.config.enabled, wrap.config.backing = True, "solid"
    scene = MaskScene()
    frames = []
    for t in (None, 55.24, 2.34, 87.14):  # burst times picked by eye at default intensity
        f = scene.render(720, 720, "x", "mask", theme, 1.0, Pose(0.18, 0.05), px=2.0)
        c = np.zeros((720, 720, 4), np.uint8)
        c[f.y:f.y + f.image.shape[0], f.x:f.x + f.image.shape[1]] = f.image
        if t is not None:
            ys, xs = np.nonzero(c[..., 3] > 8)
            glitch.apply(c, (xs.min(), ys.min(), xs.max(), ys.max()), t, theme)
        frames.append(cv2.resize(straight_rgba(c), (480, 480), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(OUT / "glitch-sprite.webp"), np.hstack(frames), [cv2.IMWRITE_WEBP_QUALITY, 90])


def screenshot() -> None:
    """The free app's main window on the MASK tab. Uses throwaway settings, never yours."""
    from PySide6.QtCore import QEventLoop, QSettings, QTimer
    from PySide6.QtWidgets import QApplication

    import n0face.app.window as window_module
    from n0face.app.main import build_window

    ini = Path(tempfile.mkdtemp()) / "n0face.ini"
    seed = QSettings(str(ini), QSettings.Format.IniFormat)
    for key, value in {"tab": 1, "theme": "toxic", "mode": "mask", "effect": "x", "wrap": True,
                       "backing": "solid", "auto_rotate": True, "reduce_motion": False}.items():
        seed.setValue(key, value)
    seed.sync()
    window_module.QSettings = lambda *_: QSettings(str(ini), QSettings.Format.IniFormat)

    app = QApplication.instance() or QApplication(sys.argv)
    window, detector = build_window(load_plugins=False)
    window.resize(1500, 940)
    window.show()
    loop = QEventLoop()
    QTimer.singleShot(2500, loop.quit)
    loop.exec()
    window.grab().toImage().save(str(OUT / "app.png"))
    shot = cv2.imread(str(OUT / "app.png"))
    cv2.imwrite(str(OUT / "app.webp"), shot, [cv2.IMWRITE_WEBP_QUALITY, 90])
    (OUT / "app.png").unlink()
    window.close()
    detector.close()


def fonts_and_icons() -> None:
    from fontTools.ttLib import TTFont

    font = TTFont(ROOT / "assets" / "fonts" / "SedgwickAveDisplay-Regular.ttf")
    font.flavor = "woff"
    font.save(OUT / "sedgwick.woff")
    shutil.copy(ROOT / "assets" / "fonts" / "SedgwickAveDisplay-OFL.txt", OUT / "sedgwick-OFL.txt")
    icon = cv2.imread(str(ROOT / "assets" / "n0face.png"), cv2.IMREAD_UNCHANGED)
    for px, name in ((32, "favicon-32.png"), (180, "apple-touch-icon.png")):
        cv2.imwrite(str(OUT / name), cv2.resize(icon, (px, px), interpolation=cv2.INTER_AREA))


def wordmark() -> None:
    """The logo's "n0face" letters on transparency (the logo itself is on black), plus the
    1200x630 social preview."""
    logo = cv2.imread(str(ROOT / "assets" / "logo.png"))
    im = logo[85:250, 596:1196].astype(np.float32)
    a = im.max(axis=2)
    # Drop the speed streaks: they are thin horizontal lines, the letters are not.
    keep = cv2.morphologyEx((a > 40).astype(np.uint8), cv2.MORPH_OPEN, np.ones((11, 1), np.uint8))
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    keep[:, :6] = keep[:, -8:] = 0
    a = (np.clip((a - 14) / (255 - 14), 0, 1) * keep)[..., None]
    rgb = np.where(a > 0, np.clip(im / np.maximum(a, 1e-3), 0, 255), 0)
    cv2.imwrite(str(OUT / "wordmark.png"), np.dstack([rgb, a * 255]).astype(np.uint8),
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

    og = np.full((630, 1200, 3), (6, 8, 5), np.uint8)
    for x in range(0, 1200, 40):
        og[:, x] = (16, 22, 14)
    for y in range(0, 630, 40):
        og[y, :] = (16, 22, 14)
    glow = np.zeros_like(og, np.float32)
    cv2.ellipse(glow, (600, 315), (520, 220), 0, 0, 360, (43, 255, 124), -1)
    og = np.clip(og + cv2.GaussianBlur(glow, (0, 0), 120) * 0.10, 0, 255).astype(np.uint8)
    w = 1080
    h = int(logo.shape[0] * w / logo.shape[1])
    small = cv2.resize(logo, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    alpha = np.clip(small.max(axis=2, keepdims=True) / 60, 0, 1)
    y0, x0 = (630 - h) // 2, (1200 - w) // 2
    roi = og[y0:y0 + h, x0:x0 + w].astype(np.float32)
    og[y0:y0 + h, x0:x0 + w] = (small * alpha + roi * (1 - alpha)).astype(np.uint8)
    cv2.imwrite(str(OUT / "og.png"), og, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    face_json()
    wordmark()
    mask_renders()
    glitch_sprite()
    fonts_and_icons()
    screenshot()
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
