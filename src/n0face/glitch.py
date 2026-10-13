"""GLITCH: short bursts of digital distortion over the masked face (mask mode).

Most frames are untouched. Every couple of seconds a burst tears the mask: slices jump
sideways, the image warps in a wave, color channels split, blocks smear and bright streaks
flash, plus the odd single-frame hiccup between bursts.

Privacy: it only moves and recolors pixels of the already-masked output, around the face.
It never reads the camera frame, so it can't reveal anything the mask hid.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from n0face.theme import Theme

WINDOW = 1.7  # seconds at FREQUENCY 1x; at most one burst per window


@dataclass
class GlitchConfig:
    enabled: bool = False
    intensity: float = 1.0  # 0.25 .. 2: how hard a burst tears
    frequency: float = 1.0  # 0.25 .. 3: how often bursts come
    length: float = 1.0  # 0.5 .. 2: how long each burst lasts


config = GlitchConfig()


def strength(t: float) -> float:
    """How hard the glitch hits at time `t`: 0 most of the time, up to INTENSITY in a burst."""
    window = WINDOW / max(config.frequency, 0.05)
    w = math.floor(t / window)
    r = np.random.default_rng(w * 7919 + 13)
    if r.random() < 0.7:
        dur = min(r.uniform(0.08, 0.45) * config.length, 0.9 * window)
        start = r.uniform(0, window - dur)
        x = t - w * window - start
        if 0 <= x < dur:
            return float(config.intensity * r.uniform(0.55, 1.0) * (1 - 0.5 * x / dur))
    frame = np.random.default_rng(int(t * 30) * 31 + 7)
    return 0.3 * config.intensity if frame.random() < min(0.02 * config.frequency, 0.2) else 0.0


def apply(img: np.ndarray, box: tuple[int, int, int, int], t: float, theme: Theme) -> None:
    """Glitch `img` (BGR, or premultiplied BGRA) in place around `box` (x0, y0, x1, y1)."""
    k = strength(t)
    if k <= 0:
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    bw = max(x1 - x0, 1)
    pad = int(0.35 * bw)  # tears run past the face
    X0, X1, Y0, Y1 = max(x0 - pad, 0), min(x1 + pad, w), max(y0, 0), min(y1, h)
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        return
    roi = img[Y0:Y1, X0:X1]
    rh, rw = roi.shape[:2]
    rng = np.random.default_rng(int(t * 30) * 104729 + 1)

    # Wave warp: rows bend in a sine with a jittery phase.
    if rng.random() < 0.6:
        ys = np.arange(rh, dtype=np.float32)
        wave = k * 0.05 * bw * np.sin(ys / rng.uniform(8, 40) + rng.uniform(0, 6.3))
        wave += (rng.random(rh // 4 + 1).repeat(4)[:rh] - 0.5) * k * 0.02 * bw
        mapx = np.arange(rw, dtype=np.float32)[None] + wave[:, None].astype(np.float32)
        mapy = np.empty((rh, rw), np.float32)
        mapy[...] = ys[:, None]
        roi[...] = cv2.remap(np.ascontiguousarray(roi), mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # Slice displacement: horizontal bands jump sideways.
    for _ in range(int(rng.integers(2, 4 + int(8 * k)))):
        y = int(rng.integers(0, rh))
        band = slice(y, min(y + int(rng.integers(2, max(int(rh * 0.07 * k), 2) + 3)), rh))
        roi[band] = np.roll(roi[band], int(rng.normal(0, 0.16 * bw * k)), axis=1)

    # Datamosh blocks: chunks copied from elsewhere in the region.
    for _ in range(int(rng.integers(0, 2 + int(5 * k)))):
        bwid, bhei = int(rng.uniform(0.08, 0.3) * bw), int(rng.uniform(0.02, 0.08) * bw) + 1
        if bwid >= rw or bhei >= rh:
            continue
        sx, sy = int(rng.integers(0, rw - bwid)), int(rng.integers(0, rh - bhei))
        dx, dy = int(rng.integers(0, rw - bwid)), int(rng.integers(0, rh - bhei))
        roi[dy : dy + bhei, dx : dx + bwid] = roi[sy : sy + bhei, sx : sx + bwid].copy()

    # RGB split: blue and red pulled apart.
    shift = int(round((1 + 5 * k) * bw / 300))
    if 0 < shift < rw:
        roi[:, :-shift, 0] = roi[:, shift:, 0]
        roi[:, shift:, 2] = roi[:, :-shift, 2]

    # Streaks: thin bright dashes in the theme colors.
    colors = [theme.primary, theme.accent, (140, 255, 120)]
    extra = (255,) if img.shape[2] == 4 else ()
    for _ in range(int(rng.integers(4, 8 + int(22 * k)))):
        length, thick = max(int(rng.uniform(0.03, 0.35) * bw), 1), int(rng.integers(1, 3))
        sx, sy = int(rng.integers(0, max(rw - length, 1))), int(rng.integers(0, rh))
        c = colors[int(rng.choice(3, p=(0.6, 0.15, 0.25)))]
        bright = rng.uniform(0.5, 1.0)
        layer = roi[sy : sy + thick, sx : sx + length]
        layer[...] = cv2.add(layer, np.full(layer.shape, tuple(int(v * bright) for v in c) + extra, np.uint8))
