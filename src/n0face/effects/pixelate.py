"""Coarse pixelation with color quantization and noise.

This is the one effect built from your actual face pixels, so it is the weakest option:
pixelated faces can sometimes be partially recovered or matched. It exists because it is the
familiar "censored" look; prefer glitch/static when the goal is concealment.
"""

import cv2
import numpy as np

from n0face.effects.base import Effect

BLOCKS_ACROSS = 6
COLOR_LEVELS = 6


class PixelateEffect(Effect):
    name = "pixelate"
    uses_camera_pixels = True

    def apply(self, frame, mask, faces, t, cover=None, blur=False):
        out = frame.copy()
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return out
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        roi = frame[y0:y1, x0:x1]
        h, w = roi.shape[:2]

        block = max(w // BLOCKS_ACROSS, 8)
        small = cv2.resize(roi, (max(w // block, 1), max(h // block, 1)), interpolation=cv2.INTER_AREA)
        # Quantize and jitter the few remaining colors to cut what's left of the signal.
        rng = np.random.default_rng(int(t * 30))
        step = 256 // COLOR_LEVELS
        small = (small // step) * step + rng.integers(0, step, small.shape)
        big = cv2.resize(np.clip(small, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        out[y0:y1, x0:x1] = cv2.copyTo(big, mask[y0:y1, x0:x1], out[y0:y1, x0:x1])
        return out
