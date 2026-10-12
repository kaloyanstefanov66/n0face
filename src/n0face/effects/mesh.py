"""Dark void with a glowing wireframe face mesh that follows your expressions,
overlaid with a sparse constellation of tracking nodes."""

import cv2
import numpy as np

from n0face.effects.base import Effect, add_glow, scanlines
from n0face.theme import DEFAULT_THEME, Theme

# Constellation over MediaPipe face mesh indices: forehead, temples, cheeks, eyes, nose, mouth, jaw.
NODES = (10, 109, 338, 234, 454, 33, 263, 1, 61, 291, 152, 172, 397)
LINKS = (
    (10, 109), (10, 338), (109, 234), (338, 454), (109, 33), (338, 263), (33, 1), (263, 1),
    (234, 172), (454, 397), (1, 61), (1, 291), (61, 152), (291, 152), (172, 152), (397, 152),
)
WHITE = (235, 235, 235)


class MeshEffect(Effect):
    name = "mesh"
    wrap_design = "holo"  # used when 3D wrap is on

    def __init__(self, edges: np.ndarray, theme: Theme = DEFAULT_THEME):
        self.edges = edges  # (N, 2) landmark index pairs
        self.theme = theme

    def render(self, h, w, faces, t, rng):
        base = np.full((h, w, 3), (18, 10, 6), dtype=np.uint8)
        stars = int(h * w * 0.004)  # faint starfield
        base[rng.integers(0, h, stars), rng.integers(0, w, stars)] = (90, 70, 30)
        if not faces:
            return scanlines(base, t, darkness=0.2)

        wire = tuple(int(v * 0.6) for v in self.theme.primary)
        lines = np.zeros_like(base)
        nodes = np.zeros_like(base)
        for lm in faces:
            pts = np.round(lm).astype(np.int32)
            segs = pts[self.edges].reshape(-1, 1, 2, 2)[:, 0]
            # Flicker a random subset of edges in the accent color for the glitch feel.
            hot = rng.random(len(segs)) < 0.03
            cv2.polylines(lines, list(segs[~hot]), False, wire, 1, cv2.LINE_AA)
            cv2.polylines(lines, list(segs[hot]), False, self.theme.accent, 1, cv2.LINE_AA)

            scale = max(int(np.ptp(pts[:, 0]) / 90), 2)
            for i, j in LINKS:
                cv2.line(nodes, tuple(pts[i]), tuple(pts[j]), WHITE, 1, cv2.LINE_AA)
            for i in NODES:
                cv2.circle(nodes, tuple(pts[i]), scale, self.theme.accent, -1, cv2.LINE_AA)

            # Occasional horizontal tear across the mesh.
            if rng.random() < 0.15:
                y = int(rng.uniform(0, h))
                band = slice(y, min(y + int(rng.uniform(3, 12)), h))
                lines[band] = np.roll(lines[band], int(rng.uniform(-25, 25)), axis=1)

        out = add_glow(base, lines, radius=3.0, strength=0.8)
        out = add_glow(out, nodes, radius=4.0, strength=1.2)
        return scanlines(out, t, darkness=0.2)
