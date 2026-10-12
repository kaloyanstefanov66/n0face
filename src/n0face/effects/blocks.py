"""Datamosh-style glitch: neon blocks, channel split, slice tearing and scanlines."""

import cv2
import numpy as np

from n0face.effects.base import Effect, scanlines
from n0face.theme import DEFAULT_THEME, Theme


class GlitchEffect(Effect):
    name = "glitch"

    def __init__(self, theme: Theme = DEFAULT_THEME):
        # Theme colors, then fixed yellow / white / dark navy / near black.
        self.palette = np.array(
            [theme.primary, theme.accent, (40, 240, 255), (255, 255, 255), (60, 30, 20), (20, 20, 20)],
            dtype=np.uint8,
        )

    def render(self, h, w, faces, t, rng):
        # Coarse mosaic of palette colors, biased toward dark so the bright ones pop.
        cell = max(min(h, w) // 10, 6)
        gh, gw = -(-h // cell), -(-w // cell)
        idx = rng.choice(len(self.palette), size=(gh, gw), p=[0.12, 0.1, 0.05, 0.03, 0.35, 0.35])
        img = cv2.resize(self.palette[idx], (gw * cell, gh * cell), interpolation=cv2.INTER_NEAREST)[:h, :w].copy()

        # Smaller bright blocks scattered on top.
        for _ in range(int(rng.integers(8, 20))):
            bw, bh = int(rng.integers(cell // 2 + 1, cell * 3)), int(rng.integers(2, cell))
            x, y = int(rng.integers(0, max(w - bw, 1))), int(rng.integers(0, max(h - bh, 1)))
            img[y : y + bh, x : x + bw] = self.palette[rng.integers(0, 4)]

        # Horizontal slice tearing.
        for _ in range(int(rng.integers(2, 6))):
            y = int(rng.integers(0, h))
            band = slice(y, min(y + int(rng.integers(2, max(h // 8, 3))), h))
            img[band] = np.roll(img[band], int(rng.integers(-w // 4 - 1, w // 4 + 1)), axis=1)

        # RGB channel split.
        shift = int(rng.integers(2, 8))
        b, g, r = cv2.split(img)
        img = cv2.merge([np.roll(b, -shift, axis=1), g, np.roll(r, shift, axis=1)])

        return scanlines(img, t)
