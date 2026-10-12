"""3D-wrapped masks: a mask texture laid out in the canonical face's UV space, stretched over
the live 478-point face mesh and lit from the estimated head pose.

Two interchangeable rasterizers draw the textured mesh:
- CPU (OpenCV): rasterizes a triangle-id map, then samples the texture with cv2.remap.
- GPU (OpenGL via moderngl, optional): 4x MSAA, depth test, smooth per-pixel lighting.

`config.backend` picks one: "auto" (GPU when it works, else CPU), "cpu" or "gpu". A GPU that
fails to initialize falls back to CPU and reports why in `status()`.

Privacy: everything drawn here comes from the texture, never from camera pixels. Wrapping
follows your facial geometry, so it's only used by the decorative `mask` mode.
"""

import math
import threading
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from n0face import canonical
from n0face.theme import Theme

TEX = 512  # texture resolution (square)
LIGHT = np.array([-0.35, 0.45, 1.0]) / np.linalg.norm([-0.35, 0.45, 1.0])  # x right, y up, z toward viewer
L_EYE, R_EYE = (33, 133, 159, 145), (362, 263, 386, 374)  # outer, inner, upper, lower
MOUTH_L, MOUTH_R, LIP_UP, LIP_LO = 61, 291, 13, 14


@dataclass
class WrapConfig:
    enabled: bool = False
    backend: str = "auto"  # "auto" | "cpu" | "gpu"


config = WrapConfig()
_status = {"renderer": "CPU", "detail": "not used yet"}
_gpu_failed: str | None = None
_local = threading.local()


def status() -> tuple[str, str]:
    """('GPU', 'NVIDIA GeForce ...') or ('CPU', reason)."""
    return _status["renderer"], _status["detail"]


def gpu_available() -> bool:
    try:
        import moderngl  # noqa: F401
    except ImportError:
        return False
    return _gpu_failed is None


# ---- geometry ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (468, 3) canonical, cm
    centered: np.ndarray  # vertices minus their mean
    uv: np.ndarray  # (468, 2) in [0, 1], v down
    tris: np.ndarray  # (T, 3)
    normals: np.ndarray  # (468, 3) vertex normals, canonical frame (z toward viewer)
    edges: np.ndarray


@lru_cache(maxsize=1)
def mesh() -> Mesh:
    face = canonical.load()
    v = face.vertices.astype(np.float64)
    tris = face.triangles.copy()
    n = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    if n[:, 2].mean() < 0:
        tris = tris[:, ::-1]
        n = -n
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, tris[:, k], n)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    return Mesh(v, v - v.mean(axis=0), face.uv.astype(np.float64), tris, vn, face.edges)


def face_pose(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the canonical mesh to 2D landmarks (weak perspective).

    Returns per-vertex normals (x right, y up, z toward the viewer) and a per-vertex
    'nearness' (larger = closer to the camera) for back-to-front ordering.
    """
    m = mesh()
    p = lm[:468, :2].astype(np.float64)
    pc = p - p.mean(axis=0)
    a, *_ = np.linalg.lstsq(m.centered, pc, rcond=None)  # pc ~= centered @ a, a is (3, 2)
    r0 = a[:, 0] / max(np.linalg.norm(a[:, 0]), 1e-9)
    r1 = a[:, 1] - (a[:, 1] @ r0) * r0
    r1 /= max(np.linalg.norm(r1), 1e-9)
    r2 = np.cross(r0, r1)  # image x right, y down -> z points into the screen
    rot = np.stack([r0, r1, r2])
    n = m.normals @ rot.T
    normals = np.stack([n[:, 0], -n[:, 1], -n[:, 2]], axis=1)
    near = -(m.centered @ r2)
    if normals[1, 2] < 0:  # nose must face the camera; guards against a reflected fit
        normals[:, 2] *= -1
        near = -near
    return normals, near


def expression(lm: np.ndarray) -> dict:
    """Eye openness (0-1), mouth openness (0-1) and smile (-1..1) from 2D landmarks."""
    p = lm[:, :2]

    def eye(o, i, u, d):
        ratio = np.linalg.norm(p[u] - p[d]) / max(np.linalg.norm(p[o] - p[i]), 1e-6)
        return float(np.clip((ratio - 0.08) / 0.22, 0, 1))

    width = max(np.linalg.norm(p[MOUTH_R] - p[MOUTH_L]), 1e-6)
    gap = np.linalg.norm(p[LIP_LO] - p[LIP_UP])
    center_y = (p[LIP_UP][1] + p[LIP_LO][1]) / 2
    corners_y = (p[MOUTH_L][1] + p[MOUTH_R][1]) / 2
    return {
        "eye_l": eye(*L_EYE), "eye_r": eye(*R_EYE),
        "mouth": float(np.clip((gap / width - 0.06) / 0.4, 0, 1)),
        "smile": float(np.clip((center_y - corners_y) / width * 5, -1, 1)),
    }


def animated_expression(t: float) -> dict:
    """For the 3D preview: blink every ~4 s, talk briefly every ~7 s, otherwise smile."""
    blink = 0.0 if (t % 4.0) < 0.14 else 1.0
    talk = 0.8 if (t % 7.0) < 0.6 else 0.0
    return {"eye_l": blink, "eye_r": blink, "mouth": talk, "smile": 0.7}


# ---- textures (designs) -----------------------------------------------------------------


def _uv_px(i) -> np.ndarray:
    return mesh().uv[i] * TEX


def _finish(bgr: np.ndarray, alpha: int) -> np.ndarray:
    """Contiguous BGR + constant alpha -> premultiplied BGRA texture."""
    if alpha < 255:
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha / 255)
    return cv2.merge([*cv2.split(bgr), np.full(bgr.shape[:2], alpha, np.uint8)])


def _glow(layer: np.ndarray, sigma: float = 3.0, strength: float = 1.4) -> np.ndarray:
    small = cv2.resize(layer, (TEX // 2, TEX // 2), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 2), (TEX, TEX), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(layer, 1.0, glow, strength, 0)


def _dim(color, f: float) -> tuple[int, int, int]:
    return tuple(int(c * f) for c in color)


@lru_cache(maxsize=1)
def _face_region() -> np.ndarray:
    """uint8 mask of the face area in UV space."""
    m = mesh()
    out = np.zeros((TEX, TEX), np.uint8)
    for tri in np.round(m.uv[m.tris] * TEX).astype(np.int32):
        cv2.fillConvexPoly(out, tri, 255)
    return cv2.dilate(out, np.ones((5, 5), np.uint8))


def _eye_centers() -> list[np.ndarray]:
    return [(_uv_px(o) + _uv_px(i)) / 2 for o, i, _, _ in (L_EYE, R_EYE)]


def _iod() -> float:
    a, b = _eye_centers()
    return float(np.linalg.norm(a - b))


@lru_cache(maxsize=16)
def _led_base(theme: Theme) -> tuple[np.ndarray, np.ndarray]:
    pitch = _iod() / 10
    xs = np.arange(pitch / 2, TEX, pitch)
    gx, gy = np.meshgrid(xs, xs)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    region = _face_region()
    pts = pts[region[pts[:, 1].astype(int).clip(0, TEX - 1), pts[:, 0].astype(int).clip(0, TEX - 1)] > 0]
    img = np.full((TEX, TEX, 3), (14, 14, 16), np.uint8)
    r = max(int(pitch * 0.34), 1)
    for x, y in np.round(pts).astype(int).tolist():
        cv2.circle(img, (x, y), r, (48, 40, 34), -1, cv2.LINE_AA)
    return img, pts


def tex_led(theme: Theme, t: float, ex: dict) -> np.ndarray:
    base, pts = _led_base(theme)
    pitch = _iod() / 10
    x, y = pts[:, 0], pts[:, 1]
    lit = np.zeros(len(pts), bool)
    line = 0.6 * pitch
    a_eye = 0.3 * _iod()
    for c, key in zip(_eye_centers(), ("eye_l", "eye_r")):
        b = max(a_eye * 0.8 * ex[key], line)
        lit |= ((x - c[0]) / a_eye) ** 2 + ((y - c[1]) / b) ** 2 <= 1
    ml, mr = _uv_px(MOUTH_L), _uv_px(MOUTH_R)
    cx, cy = (ml[0] + mr[0]) / 2, (_uv_px(LIP_UP)[1] + _uv_px(LIP_LO)[1]) / 2
    a = abs(mr[0] - ml[0]) / 2 * 1.1
    if ex["mouth"] > 0.25:
        b = max(a * 0.6 * ex["mouth"], line)
        lit |= ((x - cx) / (a * 0.6)) ** 2 + ((y - cy) / b) ** 2 <= 1
    else:
        u = np.clip((x - cx) / a, -1, 1)
        curve = cy + 0.28 * _iod() * ex["smile"] * (1 - u**2) - 0.08 * _iod() * ex["smile"]
        lit |= (np.abs(x - cx) <= a) & (np.abs(y - curve) <= line)
    light = np.zeros((TEX, TEX, 3), np.uint8)
    color = theme.accent if ex["mouth"] > 0.25 else theme.primary
    r = max(int(pitch * 0.34), 1)
    for px, py in np.round(pts[lit]).astype(int).tolist():
        cv2.circle(light, (px, py), r, color, -1, cv2.LINE_AA)
    return _finish(cv2.add(base, _glow(light)), 255)


@lru_cache(maxsize=16)
def _holo_base(theme: Theme) -> np.ndarray:
    from n0face.effects.mesh import LINKS, NODES

    m = mesh()
    lines = np.zeros((TEX, TEX, 3), np.uint8)
    segs = np.round(m.uv[m.edges] * TEX).astype(np.int32)
    cv2.polylines(lines, list(segs), False, _dim(theme.primary, 0.7), 1, cv2.LINE_AA)
    pts = np.round(m.uv * TEX).astype(int)
    for i, j in LINKS:
        cv2.line(lines, tuple(pts[i]), tuple(pts[j]), (170, 170, 170), 1, cv2.LINE_AA)
    for i in NODES:
        cv2.circle(lines, tuple(pts[i]), 4, theme.accent, -1, cv2.LINE_AA)
    return cv2.add(np.full((TEX, TEX, 3), (22, 12, 6), np.uint8), _glow(lines, 2.5, 1.0))


def tex_holo(theme: Theme, t: float, ex: dict) -> np.ndarray:
    band = np.zeros((TEX, TEX, 3), np.uint8)
    y = int((t * 0.45 % 1.0) * TEX)
    cv2.line(band, (0, y), (TEX, y), theme.primary, 3)
    return _finish(cv2.add(_holo_base(theme), _glow(band, 4, 1.6)), 255)


@lru_cache(maxsize=16)
def _circuit_base(theme: Theme) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(11)
    region = _face_region()
    keep_out = np.zeros((TEX, TEX), np.uint8)
    for c in _eye_centers():
        cv2.ellipse(keep_out, tuple(int(v) for v in c), (int(_iod() * 0.42), int(_iod() * 0.22)), 0, 0, 360, 255, -1)
    step = 14
    img = np.full((TEX, TEX, 3), (12, 10, 9), np.uint8)
    traces = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    for _ in range(90):
        p = rng.integers(0, TEX // step, 2) * step
        if region[p[1], p[0]] == 0 or keep_out[p[1], p[0]]:
            continue
        path = [p.copy()]
        d = dirs[rng.integers(0, 4)]
        for _ in range(int(rng.integers(3, 9))):
            if rng.random() < 0.35:
                d = dirs[rng.integers(0, 8)]
            nxt = path[-1] + np.array(d) * step
            if not (0 <= nxt[0] < TEX and 0 <= nxt[1] < TEX) or region[nxt[1], nxt[0]] == 0 or keep_out[nxt[1], nxt[0]]:
                break
            path.append(nxt)
        if len(path) < 2:
            continue
        poly = np.array(path, np.int32)
        traces.append(poly)
        cv2.polylines(img, [poly], False, _dim(theme.primary, 0.42), 2, cv2.LINE_AA)
        cv2.rectangle(img, tuple(poly[0] - 3), tuple(poly[0] + 3), _dim(theme.primary, 0.7), -1)
        cv2.circle(img, tuple(int(v) for v in poly[-1]), 4, _dim(theme.primary, 0.7), 1, cv2.LINE_AA)
    return img, traces


def _eye_slits(light: np.ndarray, ex: dict, color, scale: float = 1.0) -> None:
    for c, key in zip(_eye_centers(), ("eye_l", "eye_r")):
        w = int(_iod() * 0.3 * scale)
        h = max(int(_iod() * 0.13 * ex[key] * scale), 2)
        cv2.ellipse(light, tuple(int(v) for v in c), (w, h), 0, 0, 360, color, -1, cv2.LINE_AA)


def _mouth_line(light: np.ndarray, ex: dict, color) -> None:
    ml, mr = _uv_px(MOUTH_L), _uv_px(MOUTH_R)
    cx, cy = (ml[0] + mr[0]) / 2, (_uv_px(LIP_UP)[1] + _uv_px(LIP_LO)[1]) / 2
    a = abs(mr[0] - ml[0]) / 2
    if ex["mouth"] > 0.25:
        cv2.ellipse(light, (int(cx), int(cy)), (int(a * 0.6), max(int(a * 0.5 * ex["mouth"]), 2)), 0, 0, 360, color, 2, cv2.LINE_AA)
    else:
        xs = np.linspace(-1, 1, 24)
        ys = cy + 0.22 * _iod() * ex["smile"] * (1 - xs**2) - 0.07 * _iod() * ex["smile"]
        pts = np.stack([cx + xs * a, ys], axis=1).astype(np.int32)
        cv2.polylines(light, [pts], False, color, 3, cv2.LINE_AA)


def tex_circuit(theme: Theme, t: float, ex: dict) -> np.ndarray:
    base, traces = _circuit_base(theme)
    light = np.zeros((TEX, TEX, 3), np.uint8)
    for k, poly in enumerate(traces[::3]):
        seg_len = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        total = seg_len.sum()
        s = ((t * 90 + k * 37) % (total + 60)) - 30  # pulses run along the traces
        if 0 <= s <= total:
            i = int(np.searchsorted(np.cumsum(seg_len), s))
            i = min(i, len(seg_len) - 1)
            before = np.cumsum(seg_len)[i] - seg_len[i]
            f = (s - before) / max(seg_len[i], 1e-6)
            p = poly[i] + (poly[i + 1] - poly[i]) * f
            cv2.circle(light, (int(p[0]), int(p[1])), 4, theme.primary, -1, cv2.LINE_AA)
    _eye_slits(light, ex, theme.primary)
    _mouth_line(light, ex, theme.accent if ex["mouth"] > 0.25 else theme.primary)
    return _finish(cv2.add(base, _glow(light, 3, 1.6)), 255)


@lru_cache(maxsize=16)
def _hex_base(theme: Theme) -> tuple[np.ndarray, np.ndarray]:
    r = 17
    w, h = math.sqrt(3) * r, 1.5 * r
    centers = [
        (col * w + (row % 2) * w / 2, row * h)
        for row in range(int(TEX / h) + 2) for col in range(int(TEX / w) + 2)
    ]
    centers = np.array(centers)
    region = _face_region()
    inside = region[centers[:, 1].astype(int).clip(0, TEX - 1), centers[:, 0].astype(int).clip(0, TEX - 1)] > 0
    centers = centers[inside]
    img = np.full((TEX, TEX, 3), (14, 11, 9), np.uint8)
    angles = np.radians(np.arange(6) * 60 + 30)
    for cx, cy in centers:
        poly = np.stack([cx + (r - 2) * np.cos(angles), cy + (r - 2) * np.sin(angles)], axis=1).astype(np.int32)
        cv2.polylines(img, [poly], True, _dim(theme.primary, 0.38), 1, cv2.LINE_AA)
    return img, centers


def tex_hex(theme: Theme, t: float, ex: dict) -> np.ndarray:
    base, centers = _hex_base(theme)
    light = np.zeros((TEX, TEX, 3), np.uint8)
    r = 17
    angles = np.radians(np.arange(6) * 60 + 30)
    rng = np.random.default_rng(int(t * 4))
    wave = (t * 0.35 % 1.0) * TEX
    hot = (rng.random(len(centers)) < 0.05) | (np.abs(centers[:, 1] - wave) < r)
    for (cx, cy), on in zip(centers, hot):
        if on:
            poly = np.stack([cx + (r - 3) * np.cos(angles), cy + (r - 3) * np.sin(angles)], axis=1).astype(np.int32)
            cv2.fillConvexPoly(light, poly, _dim(theme.primary, 0.35))
    _eye_slits(light, ex, theme.primary, 1.1)
    _mouth_line(light, ex, theme.accent if ex["mouth"] > 0.25 else theme.primary)
    return _finish(cv2.add(base, _glow(light, 3, 1.3)), 255)


DESIGNS = {"led": tex_led, "holo": tex_holo, "circuit": tex_circuit, "hex": tex_hex}


# ---- rasterizers ------------------------------------------------------------------------


def _visible(tris, normals, near):
    tn = normals[tris].mean(axis=1)
    idx = np.nonzero(tn[:, 2] > 0.02)[0]
    order = idx[np.argsort(near[tris[idx]].mean(axis=1))]  # far first
    return order, tn


def _roi(pts: np.ndarray, h: int, w: int):
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _composite(canvas: np.ndarray, x0: int, y0: int, premul: np.ndarray) -> None:
    """Premultiplied BGRA `premul` over `canvas` (BGR, or premultiplied BGRA) at (x0, y0)."""
    h, w = premul.shape[:2]
    roi = canvas[y0 : y0 + h, x0 : x0 + w]
    pb, pg, pr, pa = cv2.split(premul)
    inv = cv2.subtract(255, pa)
    chans = cv2.split(roi)
    out = [cv2.add(p, cv2.multiply(c, inv, scale=1 / 255)) for p, c in zip((pb, pg, pr), chans[:3])]
    if canvas.shape[2] == 4:
        out.append(cv2.add(pa, cv2.multiply(chans[3], inv, scale=1 / 255)))
    roi[...] = cv2.merge(out)


class CpuRasterizer:
    name = "CPU"

    def draw(self, canvas, pts, uv, normals, near, tris, texture, rim_color):
        h, w = canvas.shape[:2]
        order, tn = _visible(tris, normals, near)
        if len(order) == 0:
            return
        tri_pts = pts[tris[order]]
        roi = _roi(tri_pts.reshape(-1, 2), h, w)
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        rh, rw = y1 - y0, x1 - x0

        # Screen -> texture affine per triangle: [x, y, 1] @ A = (u, v) in texture pixels.
        src = np.concatenate([tri_pts, np.ones((len(order), 3, 1))], axis=2)
        det = np.linalg.det(src)
        ok = np.abs(det) > 1e-6
        order, tri_pts, src = order[ok], tri_pts[ok], src[ok]
        coef = np.linalg.inv(src) @ (uv[tris[order]] * TEX)

        idmap = np.full((rh, rw), -1, np.int32)
        offset = np.array([x0, y0])
        for k, tri in enumerate(np.round(tri_pts - offset).astype(np.int32)):
            cv2.fillConvexPoly(idmap, tri, k)
        ys, xs = np.nonzero(idmap >= 0)
        if len(xs) == 0:
            return
        ids = idmap[ys, xs]
        X, Y = (xs + x0).astype(np.float32), (ys + y0).astype(np.float32)
        c = coef.astype(np.float32)
        mapx = np.full((rh, rw), -1, np.float32)
        mapy = np.full((rh, rw), -1, np.float32)
        mapx[ys, xs] = X * c[ids, 0, 0] + Y * c[ids, 1, 0] + c[ids, 2, 0]
        mapy[ys, xs] = X * c[ids, 0, 1] + Y * c[ids, 1, 1] + c[ids, 2, 1]
        sampled = cv2.remap(texture, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        n = tn[order]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        shade = (255 * (0.38 + 0.62 * np.clip(n @ LIGHT, 0, 1))).astype(np.uint8)
        rim = (255 * np.clip(1 - n[:, 2], 0, 1) ** 2 * 0.9).astype(np.uint8)
        sh = np.zeros((rh, rw), np.uint8)
        rm = np.zeros((rh, rw), np.uint8)
        sh[ys, xs] = shade[ids]
        rm[ys, xs] = rim[ids]
        b, g, r, a = cv2.split(sampled)
        rim_a = cv2.multiply(rm, a, scale=1 / 255)
        out = [
            cv2.min(cv2.add(cv2.multiply(ch, sh, scale=1 / 255), cv2.multiply(rim_a, float(col) / 255)), a)
            for ch, col in zip((b, g, r), rim_color)
        ]
        _composite(canvas, x0, y0, cv2.merge([*out, a]))


VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
in vec3 in_normal;
in float in_near;
uniform vec2 size;
uniform vec2 near_range;
out vec2 v_uv;
out vec3 v_n;
void main() {
    vec2 ndc = vec2(in_pos.x / size.x * 2.0 - 1.0, 1.0 - in_pos.y / size.y * 2.0);
    float d = (in_near - near_range.x) / max(near_range.y - near_range.x, 1e-6);
    gl_Position = vec4(ndc, 0.98 - 1.96 * d, 1.0);
    v_uv = in_uv;
    v_n = in_normal;
}
"""

FRAG = """
#version 330
uniform sampler2D tex;
uniform vec3 light;
uniform vec3 rim_color;
in vec2 v_uv;
in vec3 v_n;
out vec4 color;
void main() {
    vec3 n = normalize(v_n);
    vec4 t = texture(tex, v_uv);
    float shade = 0.38 + 0.62 * max(dot(n, light), 0.0);
    float rim = pow(clamp(1.0 - n.z, 0.0, 1.0), 2.0) * 0.9;
    vec3 rgb = min(t.rgb * shade + rim_color * rim * t.a, vec3(t.a));
    color = vec4(rgb, t.a);
}
"""


class GpuRasterizer:
    """OpenGL rasterizer. One instance (and GL context) per thread."""

    name = "GPU"

    def __init__(self):
        import moderngl

        self.mgl = moderngl
        self.ctx = moderngl.create_standalone_context()
        self.renderer = self.ctx.info.get("GL_RENDERER", "OpenGL")
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.prog["light"].value = tuple(LIGHT)
        self.tex = self.ctx.texture((TEX, TEX), 4)
        self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.vbo = self.ctx.buffer(reserve=468 * 8 * 4)
        self.ibo = self.ctx.buffer(reserve=1000 * 3 * 4)
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f 3f 1f", "in_pos", "in_uv", "in_normal", "in_near")], self.ibo
        )
        self.samples = min(4, self.ctx.max_samples)
        self._fb_size = (0, 0)

    def _framebuffers(self, w: int, h: int):
        need = (max(64, -(-w // 64) * 64), max(64, -(-h // 64) * 64))
        if need != self._fb_size:
            self._fb_size = need
            ctx = self.ctx
            self.msaa = ctx.framebuffer(
                color_attachments=[ctx.renderbuffer(need, 4, samples=self.samples)],
                depth_attachment=ctx.depth_renderbuffer(need, samples=self.samples),
            )
            self.resolved = ctx.framebuffer(color_attachments=[ctx.renderbuffer(need, 4)])
        return self.msaa, self.resolved

    def draw(self, canvas, pts, uv, normals, near, tris, texture, rim_color):
        h, w = canvas.shape[:2]
        order, _ = _visible(tris, normals, near)
        if len(order) == 0:
            return
        roi = _roi(pts[tris[order]].reshape(-1, 2), h, w)
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        rw, rh = x1 - x0, y1 - y0
        msaa, resolved = self._framebuffers(rw, rh)

        self.tex.write(np.ascontiguousarray(texture).tobytes())
        self.tex.build_mipmaps()
        verts = np.zeros((468, 8), np.float32)
        verts[:, 0:2] = pts[:468] - (x0, y0)
        verts[:, 2:4] = uv
        verts[:, 4:7] = normals
        verts[:, 7] = near
        self.vbo.write(verts.tobytes())
        self.ibo.orphan(len(order) * 12)
        self.ibo.write(tris[order].astype(np.int32).tobytes())
        self.prog["size"].value = (float(rw), float(rh))
        self.prog["near_range"].value = (float(near.min()), float(near.max()))
        self.prog["rim_color"].value = tuple(c / 255 for c in rim_color)

        msaa.use()
        self.ctx.viewport = (0, 0, rw, rh)
        msaa.clear(0, 0, 0, 0, depth=1.0)
        self.ctx.enable(self.mgl.DEPTH_TEST)
        self.tex.use(0)
        self.vao.render(self.mgl.TRIANGLES, vertices=len(order) * 3)
        self.ctx.copy_framebuffer(resolved, msaa)
        raw = resolved.read(viewport=(0, 0, rw, rh), components=4, alignment=1)
        img = np.frombuffer(raw, np.uint8).reshape(rh, rw, 4)[::-1]  # GL rows run bottom-up
        _composite(canvas, x0, y0, img)


def rasterizer():
    """The rasterizer for the current thread, honoring config.backend."""
    global _gpu_failed
    want_gpu = config.backend in ("auto", "gpu") and _gpu_failed is None
    if want_gpu:
        r = getattr(_local, "gpu", None)
        if r is None:
            try:
                r = _local.gpu = GpuRasterizer()
            except Exception as e:  # noqa: BLE001 - missing driver/package: fall back to CPU
                _gpu_failed = f"{type(e).__name__}: {e}"[:160]
        if r is not None:
            _status.update(renderer="GPU", detail=r.renderer)
            return r
    cpu = getattr(_local, "cpu", None) or CpuRasterizer()
    _local.cpu = cpu
    if config.backend == "cpu":
        detail = "selected"
    elif _gpu_failed:
        detail = f"GPU unavailable ({_gpu_failed})"
    else:
        detail = "GPU support not installed (pip install moderngl)"
    _status.update(renderer="CPU", detail=detail)
    return cpu


# ---- drawing ----------------------------------------------------------------------------


def flat_points(lm: np.ndarray) -> np.ndarray:
    """The canonical face as a flat sticker aligned to the eyes (no 3D deformation)."""
    m = mesh()
    src = np.stack([m.vertices[:, 0], -m.vertices[:, 1]], axis=1)
    s_eyes = [(src[o] + src[i]) / 2 for o, i, _, _ in (L_EYE, R_EYE)]
    d_eyes = [(lm[o, :2] + lm[i, :2]) / 2 for o, i, _, _ in (L_EYE, R_EYE)]
    sv, dv = s_eyes[1] - s_eyes[0], d_eyes[1] - d_eyes[0]
    scale = np.linalg.norm(dv) / max(np.linalg.norm(sv), 1e-9)
    ang = math.atan2(dv[1], dv[0]) - math.atan2(sv[1], sv[0])
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    mid_s, mid_d = (s_eyes[0] + s_eyes[1]) / 2, (d_eyes[0] + d_eyes[1]) / 2
    return (src - mid_s) @ rot.T * scale + mid_d


def draw_face(canvas: np.ndarray, lm: np.ndarray, design: str, theme: Theme, t: float, wrapped: bool) -> None:
    """Draw one face's mask texture onto `canvas` (in place), wrapped or flat."""
    m = mesh()
    texture = DESIGNS[design](theme, t, expression(lm))
    if wrapped:
        normals, near = face_pose(lm)
        pts = lm[:468, :2].astype(np.float64)
    else:
        pts = flat_points(lm)
        normals = np.tile([0.0, 0.0, 1.0], (468, 1))
        near = np.zeros(468)
    rasterizer().draw(canvas, pts, m.uv, normals, near, m.tris, texture, theme.primary)
