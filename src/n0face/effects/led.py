"""LED-matrix mask: a black panel of dots whose lit eyes and mouth mirror your expression.

Blinks, smiles and an open mouth are read from landmark geometry; the dot grid rotates with
head roll so the panel feels attached to your face.
"""

import cv2
import numpy as np

from n0face.effects.base import Effect, add_glow
from n0face.theme import DEFAULT_THEME, Theme

# MediaPipe face mesh landmark indices.
L_OUTER, L_INNER, L_UPPER, L_LOWER = 33, 133, 159, 145
R_INNER, R_OUTER, R_UPPER, R_LOWER = 362, 263, 386, 374
MOUTH_L, MOUTH_R, LIP_UPPER, LIP_LOWER = 61, 291, 13, 14

PANEL = (12, 12, 12)
DIM = (45, 35, 28)
GRID_PER_EYE_DISTANCE = 10


def _dim_tile() -> np.ndarray:
    tile = np.full((12, 12, 3), PANEL, dtype=np.uint8)
    cv2.circle(tile, (6, 6), 3, DIM, -1, cv2.LINE_AA)
    return tile


class LedEffect(Effect):
    name = "led"
    wrap_design = "led"  # used when 3D wrap is on

    def __init__(self, theme: Theme = DEFAULT_THEME):
        self.theme = theme

    def render(self, h, w, faces, t, rng):
        base = np.full((h, w, 3), PANEL, dtype=np.uint8)
        lit_layer = np.zeros_like(base)

        if not faces:
            return np.tile(_dim_tile(), (-(-h // 12), -(-w // 12), 1))[:h, :w].copy()

        for lm in faces:
            self._draw_face(base, lit_layer, lm, h, w, rng)
        return add_glow(base, lit_layer, radius=2.5)

    def _draw_face(self, base, lit_layer, lm, h, w, rng):
        l_eye = (lm[L_OUTER] + lm[L_INNER]) / 2
        r_eye = (lm[R_OUTER] + lm[R_INNER]) / 2
        origin = (l_eye + r_eye) / 2
        iod = float(np.linalg.norm(r_eye - l_eye))
        if iod < 4:
            return
        ex = (r_eye - l_eye) / iod
        ey = np.array([-ex[1], ex[0]])
        pitch = iod / GRID_PER_EYE_DISTANCE

        def local(p):
            d = p - origin
            return np.array([d @ ex, d @ ey])

        # Grid in face-local coordinates, mapped back into the image.
        us = np.arange(-2.6 * iod, 2.6 * iod, pitch)
        vs = np.arange(-2.8 * iod, 3.4 * iod, pitch)
        U, V = np.meshgrid(us, vs)
        U, V = U.ravel(), V.ravel()
        P = origin + U[:, None] * ex + V[:, None] * ey
        inside = (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
        U, V, P = U[inside], V[inside], P[inside]

        lit = np.zeros(len(U), dtype=bool)
        line = 0.6 * pitch

        # Eyes: fixed stylized width, height follows how open each eye is.
        for outer, inner, upper, lower in (
            (L_OUTER, L_INNER, L_UPPER, L_LOWER),
            (R_OUTER, R_INNER, R_UPPER, R_LOWER),
        ):
            c = local((lm[outer] + lm[inner]) / 2)
            openness = np.linalg.norm(lm[upper] - lm[lower]) / max(np.linalg.norm(lm[outer] - lm[inner]), 1e-6)
            a = 0.3 * iod
            b = max(a * np.clip((openness - 0.08) / 0.25, 0, 1), line)
            lit |= ((U - c[0]) / a) ** 2 + ((V - c[1]) / b) ** 2 <= 1

        # Mouth: curved line when closed (bends with a smile/frown), filled oval when open.
        cl, cr = local(lm[MOUTH_L]), local(lm[MOUTH_R])
        lu, ll = local(lm[LIP_UPPER]), local(lm[LIP_LOWER])
        a = max(abs(cr[0] - cl[0]) / 2, pitch)
        cu = (cl[0] + cr[0]) / 2
        corners_v = (cl[1] + cr[1]) / 2
        gap = ll[1] - lu[1]
        mouth_open = gap / (2 * a)
        if mouth_open > 0.12:
            cv_ = (lu[1] + ll[1]) / 2
            b = gap / 2 + line
            lit |= ((U - cu) / a) ** 2 + ((V - cv_) / b) ** 2 <= 1
        else:
            bend = 1.6 * ((lu[1] + ll[1]) / 2 - corners_v)
            x = np.clip((U - cu) / a, -1, 1)
            curve = corners_v + bend * (1 - x**2)
            lit |= (np.abs(U - cu) <= a) & (np.abs(V - curve) <= line)

        color = self.theme.accent if mouth_open > 0.35 else self.theme.primary
        radius = max(int(round(pitch * 0.35)), 1)
        for (x, y), on in zip(np.round(P).astype(int), lit):
            cv2.circle(lit_layer if on else base, (x, y), radius, color if on else DIM, -1, cv2.LINE_AA)

        # Rare row dropout, like a flaky LED driver.
        if rng.random() < 0.08:
            y = int(rng.uniform(0, h))
            lit_layer[y : y + max(radius * 2, 2)] = 0
