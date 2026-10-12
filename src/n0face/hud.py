"""In-stream biometric-scanner overlay for mask mode.

Thin target box, crosshair markers, node circles wired to an ID tag, a numeric readout column
and a glowing scan bar. Every value shown is random per session; nothing is derived from who
you are.
"""

import secrets

import cv2
import numpy as np

from n0face.theme import Theme

FONT = cv2.FONT_HERSHEY_PLAIN
WHITE = (235, 235, 235)
TAG_W = 170
TEXT_SCALE = 1.05

# MediaPipe face mesh indices.
CROSSHAIRS = (10, 33, 263, 1, 234, 454, 61, 291, 152, 172, 397, 168)
NODES = (10, 234, 1, 61, 152)
BOX_TOP, BOX_BOTTOM, BOX_LEFT, BOX_RIGHT = 10, 152, 234, 454


class Hud:
    def __init__(self, theme: Theme, session: str | None = None):
        self.theme = theme
        self.session = session or secrets.token_hex(2).upper()
        self._ids = [secrets.randbelow(900000) + 100000 for _ in range(8)]

    def draw(self, img: np.ndarray, faces: list[np.ndarray], t: float) -> None:
        rng = np.random.default_rng(int(t * 8))
        _frame_corners(img, self.theme.primary)
        _text(img, f"n0face // SESSION {self.session}", (16, 26), self.theme.primary)
        for i, lm in enumerate(faces[: len(self._ids)]):
            self._target(img, lm, self._ids[i], t, rng)

    def _target(self, img, lm, ident, t, rng):
        c, a = self.theme.primary, self.theme.accent
        x0, x1 = sorted((int(lm[BOX_LEFT, 0]), int(lm[BOX_RIGHT, 0])))
        y0, y1 = int(lm[BOX_TOP, 1]), int(lm[BOX_BOTTOM, 1])
        pad = max((x1 - x0) // 10, 4)
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 1, cv2.LINE_AA)

        size = max((x1 - x0) // 40, 3)
        for idx in CROSSHAIRS:
            px, py = np.round(lm[idx]).astype(int)
            cv2.line(img, (px - size, py), (px + size, py), WHITE, 1, cv2.LINE_AA)
            cv2.line(img, (px, py - size), (px, py + size), WHITE, 1, cv2.LINE_AA)

        # ID tag off to the side, with node circles wired into it.
        right = x1 + 14 < img.shape[1] - TAG_W - 10
        tag_x = x1 + 14 if right else x0 - 14 - TAG_W
        tag_y = y0
        anchor = (tag_x if right else tag_x + TAG_W, tag_y + 34)
        for idx in NODES:
            px, py = np.round(lm[idx]).astype(int)
            cv2.circle(img, (px, py), size * 2, WHITE, 1, cv2.LINE_AA)
            cv2.line(img, (px, py), anchor, WHITE, 1, cv2.LINE_AA)

        cv2.rectangle(img, (tag_x, tag_y), (tag_x + TAG_W, tag_y + 50), (0, 0, 0), -1)
        cv2.rectangle(img, (tag_x, tag_y), (tag_x + TAG_W, tag_y + 50), c, 1)
        _text(img, f"ID  {ident}", (tag_x + 8, tag_y + 16), WHITE)
        _text(img, f"CONF  {rng.uniform(0.9, 0.99):.6f}", (tag_x + 8, tag_y + 31), c)
        _text(img, "STATUS  MASKED", (tag_x + 8, tag_y + 45), a)

        # Readout column under the tag.
        for k in range(5):
            _text(img, f"{rng.uniform(10, 40):.6f}", (tag_x + 8, tag_y + 70 + 15 * k), c, faint=True)

        # Scan bar sweeping the box, white core with a colored glow.
        sy = y0 + int((t * 0.6 % 1.0) * (y1 - y0))
        xa, xb = max(x0, 0), min(x1, img.shape[1])
        if 0 <= sy < img.shape[0] and xb > xa:
            band = slice(max(sy - 10, 0), min(sy + 11, img.shape[0]))
            glow = np.zeros_like(img[band, xa:xb])
            cv2.line(glow, (0, sy - band.start), (xb - xa, sy - band.start), c, 3)
            glow = cv2.GaussianBlur(glow, (0, 0), 4)
            img[band, xa:xb] = cv2.add(img[band, xa:xb], glow)
            cv2.line(img, (xa, sy), (xb, sy), WHITE, 1, cv2.LINE_AA)


def _frame_corners(img, color):
    h, w = img.shape[:2]
    m, arm = 8, 28
    for cx, cy, dx, dy in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), color, 1, cv2.LINE_AA)


def _text(img, s, org, color, faint=False):
    if faint:
        color = tuple(int(v * 0.55) for v in color)
    # Outline via offset copies at the same thickness; a thicker stroke would change glyph spacing.
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, s, (x + dx, y + dy), FONT, TEXT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, TEXT_SCALE, color, 1, cv2.LINE_AA)
