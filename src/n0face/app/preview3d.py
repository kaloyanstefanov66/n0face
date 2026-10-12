"""3D preview of the masks on the canonical (generic) face.

The renderer is stateless: every call gets a `Pose`, so it can run on a background thread while
the UI owns the interaction physics. Output is a premultiplied BGRA image of just the region
the model covers (plus glow margin), and the offset to draw it at.

Rendered with OpenCV: dark panels occlude the ambient background, emissive lines and dots
glow on top of it.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from n0face import canonical
from n0face.effects import make_effects
from n0face.effects.led import L_INNER, L_OUTER, LIP_LOWER, LIP_UPPER, MOUTH_L, MOUTH_R, R_INNER, R_OUTER
from n0face.effects.mesh import LINKS, NODES
from n0face.theme import Theme

CAMERA_DIST = 5.0
LIGHT = np.array([-0.35, 0.45, 1.0]) / np.linalg.norm([-0.35, 0.45, 1.0])
DIM_DOT = (48, 40, 34)
SKIN = np.array([110, 135, 170], dtype=np.float64)


@dataclass(frozen=True)
class Pose:
    yaw: float = 0.0
    pitch: float = 0.0
    zoom: float = 1.0


@dataclass
class Frame:
    image: np.ndarray  # premultiplied BGRA, ROI only
    x: int  # ROI offset inside the full (w, h) canvas
    y: int
    width: int  # full canvas size the frame was rendered for
    height: int
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    info: dict | None = None  # hover inspection result


def _regions() -> dict[int, str]:
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as C
    except ImportError:
        return {}
    groups = (
        ("LIPS", "FACE_LANDMARKS_LIPS"),
        ("LEFT EYE", "FACE_LANDMARKS_LEFT_EYE"),
        ("RIGHT EYE", "FACE_LANDMARKS_RIGHT_EYE"),
        ("LEFT BROW", "FACE_LANDMARKS_LEFT_EYEBROW"),
        ("RIGHT BROW", "FACE_LANDMARKS_RIGHT_EYEBROW"),
        ("NOSE", "FACE_LANDMARKS_NOSE"),
        ("FACE OVAL", "FACE_LANDMARKS_FACE_OVAL"),
    )
    out: dict[int, str] = {}
    for name, attr in groups:
        for c in getattr(C, attr, []):
            out.setdefault(c.start, name)
            out.setdefault(c.end, name)
    return out


def rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rx @ ry


def focal(w: int, h: int, zoom: float) -> float:
    return 0.36 * min(w, h) * zoom * CAMERA_DIST


def project_points(p: np.ndarray, r: np.ndarray, w: int, h: int, k: float):
    pr = p @ r.T
    s = k / (CAMERA_DIST - pr[:, 2])
    return pr, np.stack([w / 2 + pr[:, 0] * s, h / 2 - pr[:, 1] * s], axis=1)


def _tri_normals(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def _premul(bgr, alpha: int) -> tuple[int, int, int, int]:
    a = alpha / 255
    return (int(bgr[0] * a), int(bgr[1] * a), int(bgr[2] * a), alpha)


class MaskScene:
    """Geometry plus rendering. Holds no interaction state."""

    def __init__(self):
        face = canonical.load()
        v = face.vertices.astype(np.float64)
        lo, hi = v.min(axis=0), v.max(axis=0)
        self._center_cm = (lo + hi) / 2
        self._scale_cm = 9.0
        self.v = (v - self._center_cm) / self._scale_cm
        tris = face.triangles.copy()
        if _tri_normals(self.v, tris)[:, 2].mean() < 0:
            tris = tris[:, ::-1]
        self.tris = tris
        self.edges = face.edges
        self.tri_normals = _tri_normals(self.v, tris)
        self.tri_centers = self.v[tris].mean(axis=1)
        vn = np.zeros_like(self.v)
        for k in range(3):
            np.add.at(vn, tris[:, k], self.tri_normals)
        self.vert_normals = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
        self._effect_cache: dict = {}
        self.regions = _regions()
        self._setup_led()
        self._setup_shell()

    # ---- setup --------------------------------------------------------------------------

    def _setup_led(self) -> None:
        v = self.v
        l_eye = (v[L_OUTER] + v[L_INNER]) / 2
        r_eye = (v[R_OUTER] + v[R_INNER]) / 2
        self.iod = float(np.linalg.norm((r_eye - l_eye)[:2]))
        self.led_pitch = self.iod / 10
        xs = np.arange(-1.0, 1.0, self.led_pitch)
        ys = np.arange(-1.1, 1.0, self.led_pitch)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        # Put each grid point on the front-most triangle under it (barycentric in the xy plane).
        a, b, c = (v[self.tris[:, k]] for k in range(3))
        best_z = np.full(len(pts), -np.inf)
        best = np.full((len(pts), 3), np.nan)
        best_tri = np.full(len(pts), -1)
        for start in range(0, len(self.tris), 128):
            sl = slice(start, start + 128)
            a2, b2, c2 = a[sl, :2], b[sl, :2], c[sl, :2]
            v0, v1 = b2 - a2, c2 - a2
            d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            d = np.where(np.abs(d) < 1e-12, 1e-12, d)
            p = pts[:, None, :] - a2[None]
            u = (p[..., 0] * v1[:, 1] - v1[:, 0] * p[..., 1]) / d
            w = (v0[:, 0] * p[..., 1] - p[..., 0] * v0[:, 1]) / d
            inside = (u >= 0) & (w >= 0) & (u + w <= 1)
            z = a[sl, 2] + u * (b[sl, 2] - a[sl, 2]) + w * (c[sl, 2] - a[sl, 2])
            z = np.where(inside, z, -np.inf)
            j = z.argmax(axis=1)
            zj = z[np.arange(len(pts)), j]
            better = zj > best_z
            best_z[better] = zj[better]
            best_tri[better] = start + j[better]
            best[better] = np.c_[pts[better], zj[better]]
        keep = best_tri >= 0
        self.led_points = best[keep]
        self.led_tri = best_tri[keep]

        # Eye and mouth geometry for the expression pattern.
        self.led_eyes = [(v[o] + v[i]) / 2 for o, i in ((L_OUTER, L_INNER), (R_OUTER, R_INNER))]
        cl, cr = v[MOUTH_L], v[MOUTH_R]
        self.led_mouth = ((cl[0] + cr[0]) / 2, (v[LIP_UPPER][1] + v[LIP_LOWER][1]) / 2, abs(cr[0] - cl[0]) / 2 * 1.1)

    def _setup_shell(self) -> None:
        lat = np.linspace(-math.radians(75), math.radians(85), 15)
        lon = np.linspace(-math.pi, math.pi, 29)
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        center = np.array([0.0, 0.08, -0.25])
        radii = np.array([1.0, 1.28, 1.05])
        unit = np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], axis=-1)
        self.shell_v = (center + unit * radii).reshape(-1, 3)
        self.shell_center = center
        nl, nm = la.shape
        quads = [
            (i * nm + j, i * nm + j + 1, (i + 1) * nm + j + 1, (i + 1) * nm + j)
            for i in range(nl - 1)
            for j in range(nm - 1)
        ]
        self.shell_quads = np.array(quads)
        centers = self.shell_v[self.shell_quads].mean(axis=1)
        normals = centers - center
        self.shell_centers = centers
        self.shell_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # ---- rendering ----------------------------------------------------------------------

    def render(
        self, w: int, h: int, effect: str, mode: str, theme: Theme, t: float,
        pose: Pose = Pose(), hover: tuple[float, float] | None = None, px: float = 1.0,
    ) -> Frame:
        r = rotation(pose.yaw, pose.pitch)
        k = focal(w, h, pose.zoom)
        lw = max(1, int(round(px)))
        geometry = self.shell_v if mode == "anonymize" else self.v
        pr_all, p2_all = project_points(geometry, r, w, h, k)

        # Region of interest: the projected model, plus room for glow and the scan bar.
        x0, y0 = p2_all.min(axis=0)
        x1, y1 = p2_all.max(axis=0)
        pad = 14 * px + 0.16 * (x1 - x0)
        rx0, ry0 = max(int(x0 - pad), 0), max(int(y0 - 14 * px), 0)
        rx1, ry1 = min(int(x1 + pad) + 1, w), min(int(y1 + 14 * px) + 1, h)
        if rx1 <= rx0 or ry1 <= ry0:
            return Frame(np.zeros((1, 1, 4), np.uint8), 0, 0, w, h, r, None)
        rw, rh = rx1 - rx0, ry1 - ry0
        shift = np.array([rx0, ry0], dtype=np.float64)
        p2 = p2_all - shift

        panel = np.zeros((rh, rw, 4), np.uint8)
        light = np.zeros((rh, rw, 3), np.uint8)
        info = None
        if mode == "anonymize":
            self._draw_shell(panel, light, pr_all, p2, r, effect, theme, t, lw)
            if hover is not None:
                info = {"hidden": True}
        else:
            design = self._texture_design(effect, theme)
            if design is not None:
                self._draw_texture(panel, pr_all, p2, r, k, shift, w, h, design, theme, t)
            elif effect == "mesh":
                self._draw_wire(light, pr_all, p2, r, theme, t, lw)
            elif effect == "led":
                self._draw_led(panel, light, pr_all, p2, r, theme, t, k, lw, shift, w, h)
            else:
                self._draw_surface(panel, pr_all, p2, r, effect, theme, t)
            if hover is not None:
                info = self._hover(p2_all, r, hover, px)

        return Frame(self._composite(panel, light, px), rx0, ry0, w, h, r, info)

    @staticmethod
    def _composite(panel: np.ndarray, light: np.ndarray, px: float) -> np.ndarray:
        """Premultiplied BGRA: emissive light (with bloom) added over the panel."""
        h, w = light.shape[:2]
        if light.any():
            small = cv2.resize(light, (max(w // 2, 1), max(h // 2, 1)), interpolation=cv2.INTER_AREA)
            glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5 * px), (w, h), interpolation=cv2.INTER_LINEAR)
            em = cv2.addWeighted(light, 1.0, glow, 1.4, 0)
        else:
            em = light
        pb, pg, pr_, pa = cv2.split(panel)
        eb, eg, er = cv2.split(em)
        alpha = cv2.max(pa, cv2.max(eb, cv2.max(eg, er)))
        b = cv2.min(cv2.add(pb, eb), alpha)
        g = cv2.min(cv2.add(pg, eg), alpha)
        r = cv2.min(cv2.add(pr_, er), alpha)
        return cv2.merge([b, g, r, alpha])

    def _facing(self, normals: np.ndarray, centers: np.ndarray, r: np.ndarray) -> np.ndarray:
        n = normals @ r.T
        view = np.array([0, 0, CAMERA_DIST]) - centers @ r.T
        return (n * view).sum(axis=1) > 0

    @staticmethod
    def _shade(normals: np.ndarray, r: np.ndarray) -> np.ndarray:
        return 0.4 + 0.6 * np.clip((normals @ r.T) @ LIGHT, 0, 1)

    def _draw_wire(self, light, pr, p2, r, theme, t, lw):
        z = pr[self.edges].mean(axis=1)[:, 2]
        zn = (z - z.min()) / max(np.ptp(z), 1e-9)
        segs = np.round(p2[self.edges]).astype(np.int32)
        band = np.minimum((zn * 4).astype(int), 3)
        for b in range(4):
            f = 0.18 + 0.27 * b
            color = tuple(int(c * f) for c in theme.primary)
            cv2.polylines(light, list(segs[band == b]), False, color, lw, cv2.LINE_AA)

        visible = (self.vert_normals @ r.T)[:, 2] > -0.1
        pts = np.round(p2).astype(np.int32).tolist()
        for i, j in LINKS:
            if visible[i] and visible[j]:
                cv2.line(light, pts[i], pts[j], (150, 150, 150), lw, cv2.LINE_AA)
        for i in NODES:
            if visible[i]:
                cv2.circle(light, pts[i], 3 * lw, theme.accent, -1, cv2.LINE_AA)

        # Scan bar sweeping the head.
        y0, y1 = p2[:, 1].min(), p2[:, 1].max()
        x0, x1 = p2[:, 0].min(), p2[:, 0].max()
        sy = int(y0 + (t * 0.45 % 1.0) * (y1 - y0))
        pad = 0.15 * (x1 - x0)
        cv2.line(light, (int(x0 - pad), sy), (int(x1 + pad), sy), theme.primary, 2 * lw, cv2.LINE_AA)

    def _fill_faces(self, img, polys, colors, order):
        """Painter's algorithm: back to front, one fill per visible polygon."""
        fill = cv2.fillConvexPoly
        polys = polys[order]
        for poly, color in zip(polys, colors[order].tolist()):
            fill(img, poly, color)

    def _visible_order(self, pr, facing):
        depth = pr[self.tris].mean(axis=1)[:, 2]
        idx = np.nonzero(facing)[0]
        return idx[np.argsort(depth[idx])]

    def _draw_led(self, panel, light, pr, p2, r, theme, t, k, lw, shift, w, h):
        facing = self._facing(self.tri_normals, self.tri_centers, r)
        shade = self._shade(self.tri_normals, r)
        g = (10 + 14 * shade).astype(np.int32)
        a = 240 / 255
        colors = np.stack([g * a, g * a, (g + 2) * a, np.full_like(g, 240)], axis=1).astype(np.int32)
        self._fill_faces(panel, np.round(p2[self.tris]).astype(np.int32), colors, self._visible_order(pr, facing))

        # Expression: blink every ~4 s, a short "o" mouth every ~7 s, otherwise a smile.
        blink = (t % 4.0) < 0.14
        talk = (t % 7.0) < 0.6
        x, y = self.led_points[:, 0], self.led_points[:, 1]
        lit = np.zeros(len(x), bool)
        line = 0.6 * self.led_pitch
        a_eye = 0.3 * self.iod
        b_eye = line if blink else a_eye * 0.8
        for c in self.led_eyes:
            lit |= ((x - c[0]) / a_eye) ** 2 + ((y - c[1]) / b_eye) ** 2 <= 1
        cx, cy, a_m = self.led_mouth
        if talk:
            lit |= ((x - cx) / (a_m * 0.55)) ** 2 + ((y - cy) / (a_m * 0.5)) ** 2 <= 1
        else:
            u = np.clip((x - cx) / a_m, -1, 1)
            curve = cy - 0.25 * self.iod * (1 - u**2) + 0.08 * self.iod
            lit |= (np.abs(x - cx) <= a_m) & (np.abs(y - curve) <= line)

        vis = facing[self.led_tri]
        _, d2 = project_points(self.led_points, r, w, h, k)
        d2 = np.round(d2 - shift).astype(np.int32)
        radius = max(int(k / CAMERA_DIST * self.led_pitch * 0.34), 1)
        color = theme.accent if talk else theme.primary
        dim = _premul(DIM_DOT, 255)
        circle = cv2.circle
        for pt in d2[vis & ~lit].tolist():
            circle(panel, pt, radius, dim, -1, cv2.LINE_AA)
        for pt in d2[vis & lit].tolist():
            circle(light, pt, radius, color, -1, cv2.LINE_AA)

    def _texture_design(self, effect: str, theme: Theme) -> str | None:
        """The wrap texture this effect shows in the preview, if it's drawn as a texture."""
        from n0face import wrap

        fx = self._effect(effect, theme)
        if fx.wrap_design and (fx.always_textured or wrap.config.enabled):
            return fx.wrap_design
        return None

    def _draw_texture(self, panel, pr, p2, r, k, shift, w, h, design, theme, t):
        """The same textured masks the live pipeline draws: wrapped on the 3D face when 3D WRAP
        is on, otherwise a flat card that turns with the head."""
        from n0face import wrap

        m = wrap.mesh()
        if wrap.config.enabled:
            pts, near = p2, pr[:, 2]
            normals = m.normals @ r.T
        else:
            card = self.v.copy()
            card[:, 2] = self.v[:, 2].max() * 0.6
            card_r, card_2d = project_points(card, r, w, h, k)
            pts, near = card_2d - shift, card_r[:, 2]
            normals = np.tile(np.array([0.0, 0.0, 1.0]) @ r.T, (len(card), 1))
        wrap.draw_design(panel, pts, normals, near, design, theme, t, lambda: wrap.animated_expression(t))

    def _effect(self, name: str, theme: Theme):
        key = (name, theme)
        if key not in self._effect_cache:
            self._effect_cache[key] = make_effects(self.edges, theme)[name]
        return self._effect_cache[key]

    def _apply_effect(self, panel, base, shade, effect, theme, t):
        """Run the real 2D effect inside the rendered silhouette, then re-apply 3D shading."""
        _, mask = cv2.threshold(shade, 0, 255, cv2.THRESH_BINARY)
        if not cv2.countNonZero(mask):
            return
        fx = self._effect(effect, theme)
        out = fx.apply(base, mask, [], t)
        if not fx.uses_camera_pixels:
            s = cv2.addWeighted(shade, 0.55, shade, 0, 0.45 * 255)  # 0.45 + 0.55 * shade
            out = cv2.multiply(out, cv2.merge([s, s, s]), scale=1 / 255)
        bgra = cv2.merge([*cv2.split(out), mask])
        cv2.copyTo(bgra, mask, panel)

    def _draw_surface(self, panel, pr, p2, r, effect, theme, t):
        """Shaded generic face (neutral tones), then the selected effect on top of it."""
        h, w = panel.shape[:2]
        facing = self._facing(self.tri_normals, self.tri_centers, r)
        shade = self._shade(self.tri_normals, r)
        # BGR skin tone in channels 0-2, shading (never 0 inside the face) in channel 3.
        colors = np.c_[SKIN[None] * shade[:, None], np.maximum(255 * shade, 1)].astype(np.int32)
        base4 = np.zeros((h, w, 4), np.uint8)
        self._fill_faces(base4, np.round(p2[self.tris]).astype(np.int32), colors, self._visible_order(pr, facing))
        b, g, rr, s = cv2.split(base4)
        self._apply_effect(panel, cv2.merge([b, g, rr]), s, effect, theme, t)

    def _draw_shell(self, panel, light, pr, p2, r, effect, theme, t, lw):
        h, w = panel.shape[:2]
        q = self.shell_quads
        facing = self._facing(self.shell_normals, self.shell_centers, r)
        shade = self._shade(self.shell_normals, r)
        quads2 = np.round(p2[q]).astype(np.int32)

        # Holographic lat/long lines on the far side for depth.
        back = tuple(int(c * 0.25) for c in theme.primary)
        cv2.polylines(light, list(quads2[~facing]), True, back, lw, cv2.LINE_AA)

        shade_map = np.zeros((h, w), np.uint8)
        depth = pr[q].mean(axis=1)[:, 2]
        idx = np.nonzero(facing)[0]
        order = idx[np.argsort(depth[idx])]
        values = np.maximum(255 * shade, 1).astype(np.int32)
        for poly, value in zip(quads2[order], values[order].tolist()):
            cv2.fillConvexPoly(shade_map, poly, value)
        self._apply_effect(panel, np.zeros((h, w, 3), np.uint8), shade_map, effect, theme, t)
        # Hide far-side lines behind the shell, then draw the near-side grid.
        light[...] = cv2.bitwise_and(light, light, mask=cv2.inRange(shade_map, 0, 0))
        front = tuple(int(c * 0.35) for c in theme.primary)
        cv2.polylines(light, list(quads2[facing]), True, front, lw, cv2.LINE_AA)

    def _hover(self, p2_full, r, hover, px):
        visible = (self.vert_normals @ r.T)[:, 2] > 0.05
        d = np.linalg.norm(p2_full - np.array(hover), axis=1)
        d[~visible] = np.inf
        i = int(d.argmin())
        if d[i] > 18 * px:
            return None
        cm = self.v[i] * self._scale_cm + self._center_cm
        return {"index": i, "pos": tuple(p2_full[i]), "cm": tuple(cm), "region": self.regions.get(i, "SURFACE")}
