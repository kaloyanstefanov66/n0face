"""3D wrap: fitting the canonical face mesh to live landmarks.

The mask designs (n0face.designs) are laid out in the canonical face's UV space and drawn on
the live 478-point face mesh, lit from the head pose estimated here. With wrap off they sit
on a flat copy of the canonical face aligned to the eyes instead (`flat_points`).

Privacy: nothing drawn comes from camera pixels. Wrapping follows your facial geometry, so
it's only used by the decorative `mask` mode.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from n0face import canonical

LIGHT = np.array([-0.35, 0.45, 1.0]) / np.linalg.norm([-0.35, 0.45, 1.0])  # x right, y up, z toward viewer
L_EYE, R_EYE = (33, 133, 159, 145), (362, 263, 386, 374)  # outer, inner, upper, lower
MOUTH_L, MOUTH_R, LIP_UP, LIP_LO = 61, 291, 13, 14


@dataclass
class WrapConfig:
    enabled: bool = False
    backing: str = "blur"  # under mask designs: "blur" (blurred face) or "solid"


config = WrapConfig()


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
