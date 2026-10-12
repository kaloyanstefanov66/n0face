"""Analog TV static with a tint and a rolling sync bar."""

import cv2
import numpy as np

from n0face.effects.base import Effect, scanlines


class StaticEffect(Effect):
    name = "static"

    def render(self, h, w, faces, t, rng):
        # Half-resolution grain, upscaled: cheaper, and chunkier static reads better on video.
        noise = rng.integers(0, 200, size=(-(-h // 2), -(-w // 2)), dtype=np.uint8)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)
        img = cv2.merge([noise, cv2.convertScaleAbs(noise, alpha=0.85), cv2.convertScaleAbs(noise, alpha=0.7)])

        bar_y = int((t * 90) % (h + 40)) - 40
        img[max(bar_y, 0) : max(bar_y + 20, 0)] //= 3

        return scanlines(img, t, darkness=0.25)
