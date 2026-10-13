"""Mask designs, drawn as vector graphics on the projected face mesh.

Each design lays out its shapes (dots, plates, traces, eye slits) once in the canonical face's
UV space and pins every point to the mesh triangle under it with fixed barycentric weights.
Per frame those points follow the projected mesh, so the shapes bend with the face, stay
crisp at any face size, dim and foreshorten with the head pose, and cost no texture work.

Every design is drawn over the same opaque, lit surface: exactly the visible triangles, so
no camera pixels show through wherever the mask covers. Lines and glow are clipped to it.
"""

import math
from dataclasses import dataclass
from functools import cached_property, lru_cache

import cv2
import numpy as np

from n0face import look
from n0face.theme import Theme
from n0face.wrap import (
    L_EYE, LIGHT, LIP_LO, LIP_UP, MOUTH_L, MOUTH_R, R_EYE, _roi, _visible, expression, face_pose, flat_points, mesh,
)

TEX = 512  # UV layout unit: design sizes are in texels of a 512 x 512 layout
SUB = 2  # sub-pixel bits for drawing
_ONE = 1 << SUB


# ---- UV anchoring -------------------------------------------------------------------------


@dataclass(frozen=True)
class _UV:
    """The canonical UV triangulation, in texels, with what anchoring and stretch need."""

    a: np.ndarray  # first corner of each triangle
    e1: np.ndarray  # edges from it
    e2: np.ndarray
    det: np.ndarray
    area: np.ndarray
    centroid: np.ndarray
    inv: np.ndarray  # inverse of [e1 e2]: UV offset -> edge coordinates
    lo: np.ndarray  # bounding boxes
    hi: np.ndarray


@lru_cache(maxsize=1)
def _uv() -> _UV:
    m = mesh()
    corners = m.uv[m.tris] * TEX
    a = corners[:, 0]
    e1, e2 = corners[:, 1] - a, corners[:, 2] - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return _UV(a, e1, e2, det, np.abs(det) / 2, corners.mean(axis=1), np.linalg.inv(np.stack([e1, e2], axis=2)),
               corners.min(axis=1), corners.max(axis=1))


@dataclass(frozen=True)
class Anchors:
    tri: np.ndarray  # (N,) triangle each point is pinned to
    bary: np.ndarray  # (N, 3) barycentric weights in that triangle


def anchor(points) -> Anchors:
    """Pin UV points (texels) to the mesh. Points just off the mesh extrapolate from the
    nearest triangle, so shapes on the border keep their outline."""
    g = _uv()
    p = np.asarray(points, np.float64).reshape(-1, 2)
    # Only triangles near the points can hold them (cheap for small per-frame shapes).
    cand = np.nonzero((g.hi >= p.min(axis=0) - 24).all(axis=1) & (g.lo <= p.max(axis=0) + 24).all(axis=1))[0]
    if len(cand) == 0:
        cand = np.arange(len(g.a))
    a, e1, e2, det, centroid = g.a[cand], g.e1[cand], g.e2[cand], g.det[cand], g.centroid[cand]
    tri = np.zeros(len(p), np.int64)
    bary = np.zeros((len(p), 3))
    for s in range(0, len(p), 256):
        q = p[s : s + 256]
        d = q[:, None, :] - a[None]
        u = (d[..., 0] * e2[:, 1] - d[..., 1] * e2[:, 0]) / det
        v = (e1[:, 0] * d[..., 1] - e1[:, 1] * d[..., 0]) / det
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        nearest = ((q[:, None, :] - centroid[None]) ** 2).sum(-1).argmin(axis=1)
        j = np.where(inside.any(axis=1), inside.argmax(axis=1), nearest)
        r = np.arange(len(q))
        uu, vv = u[r, j], v[r, j]
        tri[s : s + 256] = cand[j]
        bary[s : s + 256] = np.stack([1 - uu - vv, uu, vv], axis=1)
    return Anchors(tri, bary)


@lru_cache(maxsize=1)
def _face_region() -> np.ndarray:
    """uint8 mask of the face area in UV space."""
    m = mesh()
    out = np.zeros((TEX, TEX), np.uint8)
    for tri in np.round(m.uv[m.tris] * TEX).astype(np.int32):
        cv2.fillConvexPoly(out, tri, 255)
    return out


def _on_face(pts: np.ndarray) -> np.ndarray:
    q = np.round(pts).astype(int).clip(0, TEX - 1)
    return _face_region()[q[:, 1], q[:, 0]] > 0


def _uv_px(i) -> np.ndarray:
    return mesh().uv[i] * TEX


def _eye_centers() -> list[np.ndarray]:
    return [(_uv_px(o) + _uv_px(i)) / 2 for o, i, _, _ in (L_EYE, R_EYE)]


def _iod() -> float:
    a, b = _eye_centers()
    return float(np.linalg.norm(a - b))


def _mouth() -> tuple[float, float, float]:
    """Mouth center x, y and half-width, in texels."""
    ml, mr = _uv_px(MOUTH_L), _uv_px(MOUTH_R)
    return (ml[0] + mr[0]) / 2, (_uv_px(LIP_UP)[1] + _uv_px(LIP_LO)[1]) / 2, abs(mr[0] - ml[0]) / 2


def _ellipse(c, a: float, b: float, n: int = 28) -> np.ndarray:
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([c[0] + a * np.cos(ang), c[1] + b * np.sin(ang)], axis=1)


def _smile(ex: dict, n: int = 24, width: float = 1.0) -> np.ndarray:
    cx, cy, a = _mouth()
    xs = np.linspace(-1, 1, n)
    ys = cy + 0.22 * _iod() * ex["smile"] * (1 - xs**2) - 0.07 * _iod() * ex["smile"]
    return np.stack([cx + xs * a * width, ys], axis=1)


def _q(x: float) -> float:
    """Expression values quantized so shape anchors can be cached (steps well under a pixel)."""
    return round(float(x) * 25) / 25


# Expression-driven shapes by name; add-on designs register their own here too.
_SHAPES = {
    "mouth_ring": lambda op: _ellipse(_mouth()[:2], _mouth()[2] * 0.6, max(_mouth()[2] * 0.5 * op, 4), 28),
    "smile": lambda smile, n=24, width=1.0: _smile({"smile": smile}, n, width),
}


@lru_cache(maxsize=1024)
def _shape(kind: str, *args) -> Anchors:
    """Anchors of an expression-driven shape; args are quantized with `_q`."""
    return anchor(_SHAPES[kind](*args))


def _ink(color) -> tuple:
    """An opaque colour for the premultiplied BGRA crisp layer."""
    return (*tuple(color)[:3], 255)


def _mix(color, f: float, white: float = 0.0) -> tuple[int, int, int]:
    return tuple(int(min(c * f + (255 - c * f) * white, 255)) for c in color)


# ---- painter ------------------------------------------------------------------------------


BLUR_DARKEN = 0.85  # the blurred face is dimmed so the design on top stands out


def blur_backing(img: np.ndarray, region: np.ndarray) -> None:
    """Replace `region` (uint8 mask) of `img` with a heavy, darkened blur of itself, in place.

    It's scaled down to about 24 px across first, so eyes, nose and mouth don't survive; what's
    left is colour and rough shape under the mask design.
    """
    x, y, w, h = cv2.boundingRect(region)
    if w == 0:
        return
    roi = img[y : y + h, x : x + w]
    f = max(w, h) / 24
    small = cv2.resize(roi, (max(round(w / f), 1), max(round(h / f), 1)), interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.2), (w, h), interpolation=cv2.INTER_LINEAR)
    blurred[..., :3] = cv2.convertScaleAbs(blurred[..., :3], alpha=BLUR_DARKEN)  # alpha (BGRA) as is
    cv2.copyTo(blurred, region[y : y + h, x : x + w], roi)


class Painter:
    """One face being drawn: the surface, a crisp layer (premultiplied BGRA, so a design can
    leave the backing visible) and a half-size glow layer.

    With `blur` the surface is see-through: the canvas there already holds a blurred face
    (`blur_backing`), and only the design's own marks cover it.
    """

    def __init__(self, canvas: np.ndarray, pts, normals, near, theme: Theme, blur: bool = False):
        self.canvas, self.theme, self.blur = canvas, theme, blur
        m = mesh()
        self.order, tn = _visible(m.tris, normals, near)
        self.roi = _roi(pts[m.tris[self.order]].reshape(-1, 2), *canvas.shape[:2]) if len(self.order) else None
        if self.roi is None:
            return
        x0, y0, x1, y1 = self.roi
        self.rw, self.rh = x1 - x0, y1 - y0
        self.local = pts[:468] - (x0, y0)
        self.near = near
        self.facing = tn[:, 2] > 0.02
        n = tn / np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-9)
        self.n = n
        self.shade = 0.38 + 0.62 * np.clip(n @ LIGHT, 0, 1)
        self.rim = np.clip(1 - n[:, 2], 0, 1) ** 2 * 0.9
        tp = self.local[m.tris]
        e1, e2 = tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0]
        area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
        self.scale = np.sqrt(area / np.maximum(_uv().area, 1e-9))  # screen px per texel
        self._edges = e1, e2
        self.face_scale = float(np.median(self.scale[self.order]))
        self.surf = np.zeros((self.rh, self.rw, 4), np.uint8)
        self.small = np.zeros(((self.rh + 1) // 2, (self.rw + 1) // 2, 3), np.uint8)

    @cached_property
    def squash(self) -> np.ndarray:
        """Smallest stretch of the UV -> screen map per triangle: where the surface turns away
        from the camera it squashes, and round things drawn there should shrink."""
        jac = np.stack(self._edges, axis=2) @ _uv().inv
        tr = (jac**2).sum(axis=(1, 2))
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        return np.sqrt(np.maximum(tr / 2 - np.sqrt(np.maximum(tr**2 / 4 - det**2, 0)), 0))

    @property
    def ok(self) -> bool:
        return self.roi is not None

    def surface(self, base, extra: np.ndarray | None = None, veil: float = 0.0) -> None:
        """The visible triangles in painter's order: an opaque lit surface in `base` colour, or
        with `blur` a see-through one that only carries the rim light, `extra` (per-triangle
        primary glow) and `veil` (how much of the base colour covers the blurred face). Marks
        the area the design may draw on."""
        o = self.order
        tint = self.rim[o] if extra is None else self.rim[o] + extra[o]
        primary = np.array(self.theme.primary, np.float64)
        if self.blur:
            glow = np.clip(tint * 0.7, 0, 1)
            lit = np.array(base, np.float64) * self.shade[o, None] * veil
            colors = np.c_[np.minimum(lit + primary * glow[:, None], 255), 255 * np.maximum(glow, veil)]
        else:
            lit = np.minimum(np.array(base, np.float64) * self.shade[o, None] + primary * tint[:, None], 255)
            colors = np.c_[lit, np.full(len(o), 255.0)]
        self.cover = np.zeros((self.rh, self.rw), np.uint8)
        fill = cv2.fillConvexPoly
        tris = np.round(self.local[mesh().tris[o]]).astype(np.int32)
        for tri, color in zip(tris, colors.astype(np.int32).tolist()):
            fill(self.surf, tri, color)
            fill(self.cover, tri, 255)

    def project(self, anc: Anchors):
        """Anchored UV points -> (screen points in ROI px, visible, local px per texel)."""
        corners = self.local[mesh().tris[anc.tri]]
        return np.einsum("nk,nkd->nd", anc.bary, corners), self.facing[anc.tri], self.scale[anc.tri]

    # Drawing. Shapes are float ROI coordinates; `glow` scales what goes into the glow layer.

    @staticmethod
    def _fx(p) -> np.ndarray:
        return np.round(np.asarray(p) * _ONE).astype(np.int32)

    def lines(self, polys, color, closed=False, thickness=1, glow=0.0) -> None:
        if len(polys) == 0:
            return
        polys = [self._fx(p) for p in polys] if isinstance(polys, list) else self._fx(polys)
        thickness = max(int(round(thickness * look.look.stroke)), 1)
        cv2.polylines(self.surf, polys, closed, _ink(color), thickness, cv2.LINE_AA, SUB)
        if glow:
            cv2.polylines(self.small, polys, closed, _mix(color, glow), max(thickness // 2, 1), cv2.LINE_8, SUB + 1)

    def fill(self, poly, color, glow=0.0) -> None:
        p = self._fx(poly)
        cv2.fillPoly(self.surf, [p], _ink(color), cv2.LINE_AA, SUB)
        if glow:
            cv2.fillPoly(self.small, [p], _mix(color, glow), cv2.LINE_8, SUB + 1)

    def fill_each(self, polys, colors, alpha: float = 1.0) -> None:
        """Many small convex shapes, each its own color (no glow); `alpha` < 1 lets the backing
        show through."""
        fill = cv2.fillConvexPoly
        polys = [self._fx(q) for q in polys] if isinstance(polys, list) else self._fx(polys)
        if alpha >= 1:
            for p, c in zip(polys, colors):
                fill(self.surf, p, _ink(c), cv2.LINE_AA, SUB)
            return
        # Translucent: paint opaquely on a layer of its own, then lay it over what's there.
        layer = np.zeros_like(self.surf)
        for p, c in zip(polys, colors):
            fill(layer, p, _ink(c), cv2.LINE_AA, SUB)
        keep = cv2.subtract(255, cv2.multiply(cv2.extractChannel(layer, 3), alpha))
        keep4 = cv2.merge([keep] * 4)
        self.surf[...] = cv2.add(cv2.multiply(layer, alpha), cv2.multiply(self.surf, keep4, scale=1 / 255))

    def dots(self, centers, radii, color, glow=0.0) -> None:
        circle = cv2.circle
        c = self._fx(centers).tolist()
        r = np.maximum(np.round(np.asarray(radii) * _ONE), 1).astype(int).tolist()
        g = _mix(color, glow) if glow else None
        ink = _ink(color)
        for (x, y), rr in zip(c, r):
            circle(self.surf, (x, y), rr, ink, -1, cv2.LINE_AA, SUB)
            if g:
                circle(self.small, (x, y), max(rr, 2), g, -1, cv2.LINE_8, SUB + 1)

    def rings(self, centers, radii, color) -> None:
        circle = cv2.circle
        r = np.maximum(np.round(np.asarray(radii) * _ONE), 1).astype(int).tolist()
        ink = _ink(color)
        for (x, y), rr in zip(self._fx(centers).tolist(), r):
            circle(self.surf, (x, y), rr, ink, 1, cv2.LINE_AA, SUB)

    def finish(self) -> None:
        """Composite the design (premultiplied) over the canvas, plus glow, inside the surface."""
        rw, rh = self.rw, self.rh
        x0, y0, x1, y1 = self.roi
        dst = self.canvas[y0:y1, x0:x1]
        glow = cv2.resize(cv2.GaussianBlur(self.small, (0, 0), max(rw / 300, 1.0)), (rw, rh), interpolation=cv2.INTER_LINEAR)
        b, g, r, a = cv2.split(self.surf)
        inv = cv2.subtract(255, a)
        chans = cv2.split(dst)
        out = [cv2.add(c, cv2.multiply(d, inv, scale=1 / 255)) for c, d in zip((b, g, r), chans[:3])]
        out = cv2.addWeighted(cv2.merge(out), 1.0, glow, 0.6 * look.look.glow, 0)
        if dst.shape[2] == 4:
            out = cv2.merge([*cv2.split(out), cv2.add(a, cv2.multiply(chans[3], inv, scale=1 / 255))])
        cv2.copyTo(out, self.cover, dst)


# ---- MESH (holo): the face mesh's own wireframe --------------------------------------------


@lru_cache(maxsize=1)
def _holo_geometry():
    from n0face.effects.mesh import LINKS, NODES

    m = mesh()
    v = m.uv[m.tris].mean(axis=1)[:, 1]
    return (v - v.min()) / np.ptp(v), np.array(LINKS), np.array(NODES)


def draw_holo(pa: Painter, t: float, ex: dict) -> None:
    th = pa.theme
    scan_v, links, nodes = _holo_geometry()
    scan = np.clip(1 - np.abs(scan_v - (t * 0.45 % 1.0)) / 0.05, 0, 1)
    pa.surface((22, 12, 6), 0.4 * scan)

    vis = np.zeros(468, bool)  # vertices on a visible triangle
    vis[mesh().tris[pa.order].ravel()] = True
    edges = mesh().edges
    ok = vis[edges].all(axis=1)
    depth = pa.near[edges].mean(axis=1)
    lo, hi = (depth[ok].min(), depth[ok].max()) if ok.any() else (0.0, 1.0)
    band = np.minimum(((depth - lo) / max(hi - lo, 1e-9) * 4).astype(int), 3)
    segs = pa.local[edges]
    line = look.color("lines", th.primary)
    for b in range(4):
        pa.lines(segs[ok & (band == b)], _mix(line, 0.3 + 0.2 * b), glow=1.0)
    pa.lines(pa.local[links[vis[links].all(axis=1)]], look.color("accent", (170, 170, 170)), glow=1.0)
    nodes = nodes[vis[nodes]]
    pa.dots(pa.local[nodes], np.full(len(nodes), max(2.0, pa.rw / 110)), look.color("accent", th.accent), glow=1.0)


# ---- brush strokes (X) ---------------------------------------------------------------------

PAINT = (228, 232, 236)  # BGR, spray-paint white


def _resample(path: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(cum[-1] / step), 1) + 1
    s = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])], axis=1)


def _rough(path: np.ndarray, amp: float, seed: int) -> np.ndarray:
    """A hand-drawn version of `path`: wobbled sideways by smooth noise of about `amp` texels."""
    pts = _resample(path, 3.0)
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 1, len(pts)))
    walk = np.convolve(walk - np.linspace(walk[0], walk[-1], len(pts)), np.ones(5) / 5, mode="same")
    walk *= amp / max(np.abs(walk).max(), 1e-6)
    tangent = np.gradient(pts, axis=0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    return pts + normal * walk[:, None]


def _drips(path: np.ndarray, count: int, length: float, seed: int) -> list[np.ndarray]:
    """Paint runs hanging from points of a stroke (downwards in UV, i.e. down the face)."""
    rng = np.random.default_rng(seed + 991)
    out = []
    for i in rng.choice(len(path), size=min(count, len(path)), replace=False):
        top = path[i]
        out.append(np.array([top, top + (rng.normal(0, 0.6), length * rng.uniform(0.4, 1.0))]))
    return out


def _x_eye(c: np.ndarray, op: float, iod: float, seed: int) -> list[np.ndarray]:
    """Two crossed brush strokes over an eye; they squash as the eye closes."""
    a, b = 0.3 * iod, 0.3 * iod * max(op, 0.35)
    return [_rough(np.array([c + d0, c + d1]), 4.0, seed + j)
            for j, (d0, d1) in enumerate((((-a, -b), (a, b)), ((a, -b), (-a, b))))]


def _stitched_mouth(smile: float, mouth: float, boil: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """A wide grin seam and the stitches across it; they stretch as the mouth opens."""
    iod = _iod()
    cx, cy, a = _mouth()
    xs = np.linspace(-1, 1, 40)
    grin = 0.12 + 0.18 * max(smile, 0)
    ys = cy + iod * (grin * (1 - xs**2) - 0.06) - 0.08 * iod * xs**4
    seam = _rough(np.stack([cx + xs * a * 1.75, ys], axis=1), 2.8, 100 * boil + 50)
    gap = 0.05 * iod + 0.25 * iod * mouth
    stitches = []
    rng = np.random.default_rng(100 * boil + 77)
    for x in np.linspace(-0.85, 0.85, 11):
        i = int((x + 1) / 2 * (len(seam) - 1))
        c = seam[i] + (0, rng.normal(0, 0.8))
        half = 0.07 * iod + gap / 2
        stitches.append(_rough(np.array([c - (rng.normal(0, 2.5), half), c + (rng.normal(0, 2.5), half)]), 1.8, 100 * boil + i))
    return [seam], stitches


# Swappable X parts (look.part("eyes" / "mouth")). Add-ons may register more variants:
# eyes: (center, openness, iod, seed) -> strokes; mouths: (smile, open, boil) -> (strokes, thin strokes)
X_EYES = {"x": _x_eye}
X_MOUTHS = {"stitches": _stitched_mouth}


def _x_shapes(ex_open_l, ex_open_r, smile, mouth, boil, eyes_kind="x", mouth_kind="stitches"):
    """UV polylines for the X mask: eyes, mouth strokes and thin strokes, and their drips."""
    iod = _iod()
    eye_fn = X_EYES.get(eyes_kind, _x_eye)
    eyes, eye_drips = [], []
    for k, (c, op) in enumerate(zip(_eye_centers(), (ex_open_l, ex_open_r))):
        for j, stroke in enumerate(eye_fn(c, op, iod, 100 * boil + 10 * k)):
            eyes.append(stroke)
            eye_drips += _drips(stroke, 3 if len(stroke) > 6 else 1, 0.2 * iod, 10 * k + j)  # drips stay put while the paint boils
    seam, stitches = X_MOUTHS.get(mouth_kind, _stitched_mouth)(smile, mouth, boil)
    mouth_drips = _drips(seam[0], 4, 0.18 * iod, 60) if seam and len(seam[0]) > 4 else []
    return eyes, eye_drips, seam, stitches, mouth_drips


@lru_cache(maxsize=256)
def _x_anchors(op_l: float, op_r: float, smile: float, mouth: float, boil: int, eyes_kind: str, mouth_kind: str):
    groups = list(_x_shapes(op_l, op_r, smile, mouth, boil, eyes_kind, mouth_kind))
    flat = [p for g in groups for p in g]
    return anchor(np.concatenate(flat)), [[len(p) for p in g] for g in groups]


def _paint(pa: Painter, polys, color, width: float, glow: float = 0.0) -> None:
    """Spray-painted strokes: a solid core and a thinner, offset dry-brush pass."""
    tk = max(int(round(width * pa.face_scale)), 1)
    pa.lines(polys, color, thickness=tk, glow=glow)
    rough = [q + (0.35 * tk, -0.25 * tk) for q in polys]
    pa.lines(rough, _mix(color, 0.8), thickness=max(tk // 2, 1))


def draw_x(pa: Painter, t: float, ex: dict) -> None:
    """Graffiti purge mask: painted X eyes and a stitched-shut grin that bends with yours."""
    th = pa.theme
    pa.surface(look.color("base", (12, 12, 13)))
    boil = int(t * 6) % 3  # stop-motion wobble, like hand-drawn animation
    anc, sizes = _x_anchors(_q(ex["eye_l"]), _q(ex["eye_r"]), _q(ex["smile"]), _q(ex["mouth"]), boil,
                            look.part("eyes", "x"), look.part("mouth", "stitches"))
    p, vis, _ = pa.project(anc)
    groups, i = [], 0
    for g in sizes:
        polys = []
        for n in g:
            if vis[i : i + n].all():
                polys.append(p[i : i + n])
            i += n
        groups.append(polys)
    eyes, eye_drips, seam, stitches, mouth_drips = groups
    eye_color, mouth_color = look.color("eyes", th.primary), look.color("mouth", PAINT)
    _paint(pa, eyes, eye_color, 16, glow=1.0)
    _paint(pa, seam, mouth_color, 8)
    _paint(pa, stitches, mouth_color, 7)
    if not look.look.drips:
        return
    _paint(pa, eye_drips, eye_color, 5, glow=0.6)
    if eye_drips:
        pa.dots(np.array([d[-1] for d in eye_drips]), np.full(len(eye_drips), 3.6 * pa.face_scale), eye_color)
    _paint(pa, mouth_drips, mouth_color, 3.5)
    if mouth_drips:
        pa.dots(np.array([d[-1] for d in mouth_drips]), np.full(len(mouth_drips), 2.6 * pa.face_scale), mouth_color)


# Designs by name; add-ons register theirs here too.
DESIGNS = {"holo": draw_holo, "x": draw_x}


# ---- entry points -------------------------------------------------------------------------


def draw_design(canvas, pts, normals, near, design: str, theme: Theme, t: float, ex, blur: bool = False) -> None:
    """One design on a projected face mesh, in place. `ex` returns the expression. With `blur`
    the design's surface is see-through, for a canvas whose face is already blurred."""
    pa = Painter(canvas, pts, normals, near, theme, blur)
    if not pa.ok:
        return
    DESIGNS[design](pa, t, ex())
    pa.finish()


def draw_face(canvas: np.ndarray, lm: np.ndarray, design: str, theme: Theme, t: float, wrapped: bool, blur: bool = False) -> None:
    """Draw one face's mask onto `canvas` (BGR or premultiplied BGRA), wrapped or flat."""
    if wrapped:
        normals, near = face_pose(lm)
        pts = lm[:468, :2].astype(np.float64)
    else:
        pts = flat_points(lm)
        normals = np.tile([0.0, 0.0, 1.0], (468, 1))
        near = np.zeros(468)
    draw_design(canvas, pts, normals, near, design, theme, t, lambda: expression(lm), blur)
