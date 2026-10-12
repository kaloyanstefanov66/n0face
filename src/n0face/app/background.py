"""Ambient background: drifting accent glows, faint grid, film grain, scanlines and a vignette.

Everything is pre-composed into one cached pixmap. The static layers (grid, grain, scanlines,
vignette) are built once per window size; the glows are rendered at 1/8 resolution (they're
soft, so nothing is lost) and the composite is refreshed a few times a second. A paint is then
a single clipped pixmap blit, no matter how often the UI repaints over it.
"""

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QRadialGradient

from n0face.app.style import BG

GLOW_SCALE = 8


def _grain(w: int, h: int) -> QImage:
    rng = np.random.default_rng(3)
    a = (rng.random((h, w)) * 16).astype(np.uint8)
    v = rng.integers(0, 255, (h, w), dtype=np.uint8)
    px = (v.astype(np.uint16) * a // 255).astype(np.uint8)
    buf = np.ascontiguousarray(np.dstack([px, px, px, a]))
    return QImage(buf.data, w, h, 4 * w, QImage.Format.Format_ARGB32_Premultiplied).copy()


class Ambient:
    def __init__(self):
        self.size = QSize()
        self.static: QPixmap | None = None
        self.frame: QPixmap | None = None
        self.accent = QColor()
        self.frozen = False  # reduced motion: glows stop drifting

    def _build_static(self, size: QSize, dpr: float) -> None:
        w, h = int(size.width() * dpr), int(size.height() * dpr)
        pm = QPixmap(w, h)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.scale(dpr, dpr)
        rect = QRectF(0, 0, size.width(), size.height())
        p.setPen(QPen(QColor(255, 255, 255, 7), 1))
        step = 48
        for x in range(0, size.width(), step):
            p.drawLine(QPointF(x, 0), QPointF(x, size.height()))
        for y in range(0, size.height(), step):
            p.drawLine(QPointF(0, y), QPointF(size.width(), y))
        p.resetTransform()
        p.fillRect(QRectF(0, 0, w, h), QBrush(_grain(256, 256)))
        p.scale(dpr, dpr)
        scan = np.zeros((4, 4, 4), np.uint8)
        scan[0, :, 3] = 22
        p.fillRect(rect, QBrush(QImage(scan.data, 4, 4, 16, QImage.Format.Format_ARGB32_Premultiplied).copy()))
        v = QRadialGradient(rect.center(), math.hypot(rect.width(), rect.height()) * 0.62)
        v.setColorAt(0.55, QColor(0, 0, 0, 0))
        v.setColorAt(1.0, QColor(0, 0, 0, 170))
        p.fillRect(rect, QBrush(v))
        p.end()
        self.static = pm

    def refresh(self, size: QSize, dpr: float, accent: QColor, t: float) -> None:
        """Recompose the cached frame (call a few times per second)."""
        if size != self.size or self.static is None:
            self.size = QSize(size)
            self._build_static(size, dpr)
        if self.frozen and self.frame is not None and accent == self.accent:
            return
        self.accent = QColor(accent)
        gw, gh = max(size.width() // GLOW_SCALE, 2), max(size.height() // GLOW_SCALE, 2)
        glow = QImage(gw, gh, QImage.Format.Format_ARGB32_Premultiplied)
        glow.fill(QColor(BG))
        p = QPainter(glow)
        blobs = (
            (0.30, 0.38, 0.050, 0.0, 0.60, accent, 46),
            (0.74, 0.70, 0.037, 2.1, 0.50, accent, 30),
            (0.55, 0.12, 0.029, 4.0, 0.65, QColor(20, 90, 100), 60),
            (0.10, 0.90, 0.021, 1.0, 0.45, QColor(60, 20, 90), 40),
        )
        tt = 0.0 if self.frozen else t
        for bx, by, speed, phase, radius, color, alpha in blobs:
            cx = gw * (bx + 0.12 * math.sin(tt * speed * 2.0 + phase))
            cy = gh * (by + 0.10 * math.cos(tt * speed * 1.6 + phase))
            g = QRadialGradient(QPointF(cx, cy), max(gw, gh) * radius)
            c0, c1 = QColor(color), QColor(color)
            c0.setAlpha(alpha)
            c1.setAlpha(0)
            g.setColorAt(0.0, c0)
            g.setColorAt(1.0, c1)
            p.fillRect(QRectF(0, 0, gw, gh), QBrush(g))
        p.end()

        frame = QPixmap(self.static.size())
        p = QPainter(frame)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(0, 0, frame.width(), frame.height()), glow)
        p.drawPixmap(0, 0, self.static)
        p.end()
        frame.setDevicePixelRatio(dpr)
        self.frame = frame

    def paint(self, p: QPainter, clip: QRectF) -> None:
        if self.frame is None:
            p.fillRect(clip, QColor(BG))
            return
        dpr = self.frame.devicePixelRatio()
        src = QRectF(clip.left() * dpr, clip.top() * dpr, clip.width() * dpr, clip.height() * dpr)
        p.drawPixmap(clip, self.frame, src)
