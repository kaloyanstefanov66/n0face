"""Steadying the drawn masks over time, so they don't twitch.

Detections shake by a few pixels from frame to frame, more so for heads found by pose. These
smooth the face meshes the mask designs are drawn on, one track per person:

- `steady` (recorded videos) averages each mesh with the same person's mesh in the frames
  just before and after it. Looking both ways adds no lag.
- `Steadier` (live) can't look ahead, so it uses a One Euro filter: heavy smoothing while the
  head is still, light while it moves, so the mask doesn't trail a fast turn.

Only what is drawn is smoothed. The privacy mask is still built from the raw detections.
"""

import math

import numpy as np

from n0face.detector import Found
from n0face.heads import Head

SAME_PERSON = 0.5  # meshes in different frames are one person if their centers are this many widths apart


def _meshes(found: Found) -> list[np.ndarray]:
    return [*found.faces, *(h.landmarks for h in found.heads if not h.has_face)]


def _center_width(mesh: np.ndarray) -> tuple[np.ndarray, float]:
    return mesh[:468].mean(axis=0), max(float(np.ptp(mesh[:468, 0])), 1.0)


def _nearest(mesh: np.ndarray, others: list[np.ndarray]) -> int | None:
    if not others:
        return None
    c, w = _center_width(mesh)
    d = [float(np.linalg.norm(_center_width(o)[0] - c)) for o in others]
    j = int(np.argmin(d))
    return j if d[j] < SAME_PERSON * w else None


def steady(found: Found, neighbors: list[tuple[int, Found]], sigma: float) -> Found:
    """`found` with each mesh averaged over `neighbors` ((frame offset, Found) pairs), weighted
    by a Gaussian of `sigma` frames."""
    if not found or sigma <= 0:
        return found
    near = [(math.exp(-off * off / (2 * sigma * sigma)), _meshes(f)) for off, f in neighbors]

    def smooth(mesh: np.ndarray) -> np.ndarray:
        acc, total = mesh.astype(np.float64), 1.0
        for weight, meshes in near:
            j = _nearest(mesh, meshes)
            if j is not None:
                acc += weight * meshes[j]
                total += weight
        return (acc / total).astype(np.float32)

    heads = [h if h.has_face else Head(h.points, smooth(h.landmarks)) for h in found.heads]
    return Found([smooth(f) for f in found.faces], heads)


class Steadier:
    """Causal smoothing for live video: a One Euro filter per tracked person.

    `min_cutoff` (Hz) sets how still a still head stays; `beta` how quickly the filter loosens
    as the head moves (per face width per second).
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.8, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._tracks: list[tuple[np.ndarray, float, float]] = []  # (mesh, speed, time)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def __call__(self, found: Found, t: float) -> Found:
        tracks, prev = [], list(self._tracks)

        def step(mesh: np.ndarray) -> np.ndarray:
            j = _nearest(mesh, [m for m, _, _ in prev])
            if j is None:
                tracks.append((mesh, 0.0, t))
                return mesh
            last, speed, t0 = prev.pop(j)
            dt = max(t - t0, 1e-3)
            c, w = _center_width(mesh)
            v = float(np.linalg.norm(c - _center_width(last)[0])) / dt / w
            speed += self._alpha(self.d_cutoff, dt) * (v - speed)
            a = self._alpha(self.min_cutoff + self.beta * speed, dt)
            out = (last + a * (mesh - last)).astype(np.float32)
            tracks.append((out, speed, t))
            return out

        faces = [step(f) for f in found.faces]
        out = Found(faces, [h if h.has_face else Head(h.points, step(h.landmarks)) for h in found.heads])
        self._tracks = tracks
        return out
