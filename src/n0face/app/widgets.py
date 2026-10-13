"""Painted, spring-animated components that make up the sidebar.

Every interactive control animates hover, press and selection with springs from
`n0face.app.motion`; nothing animates while idle.
"""

import math
import time
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractButton, QGraphicsOpacityEffect, QSizePolicy, QStackedWidget, QWidget

from n0face.app import motion
from n0face.app.motion import BOUNCY, SMOOTH, SNAPPY, Spring, mix
from n0face.app.style import DIM, FAINT, LINE_STRONG, LIVE, OK, TEXT, WARN, graffiti, mono, spray_text, with_alpha

ACCENT = QColor("#28e6ff")  # replaced at runtime by set_accent()
INK = QColor("#061015")  # text/glyph color on accent fills


def set_accent(color: QColor) -> None:
    ACCENT.setRgb(color.red(), color.green(), color.blue())


def blend(a: QColor, b: QColor, t: float) -> QColor:
    t = min(max(t, 0.0), 1.0)
    return QColor(
        int(mix(a.red(), b.red(), t)), int(mix(a.green(), b.green(), t)),
        int(mix(a.blue(), b.blue(), t)), int(mix(a.alpha(), b.alpha(), t)),
    )


def brackets(p: QPainter, r: QRectF, arm: float) -> None:
    for (x, y), (dx, dy) in (
        ((r.left(), r.top()), (1, 1)), ((r.right(), r.top()), (-1, 1)),
        ((r.left(), r.bottom()), (1, -1)), ((r.right(), r.bottom()), (-1, -1)),
    ):
        p.drawLine(QPointF(x, y), QPointF(x + dx * arm, y))
        p.drawLine(QPointF(x, y), QPointF(x, y + dy * arm))


class Animated(QAbstractButton):
    """Button base with hover / press / checked springs."""

    def __init__(self, check_preset=SNAPPY):
        super().__init__()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hover = Spring(0, SNAPPY, self.update)
        self.press = Spring(0, BOUNCY, self.update)
        self.check = Spring(0, check_preset, self.update)
        self.toggled.connect(lambda on: self.check.to(1.0 if on else 0.0))

    def setChecked(self, on: bool) -> None:  # noqa: N802 - Qt naming
        was = self.isChecked()
        super().setChecked(on)
        if not self.isVisible() or was == on:
            self.check.snap(1.0 if on else 0.0)

    def enterEvent(self, e):
        self.hover.to(1)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.hover.to(0)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self.press.to(1)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        self.press.to(0)
        super().mouseReleaseEvent(e)

    def changeEvent(self, e):
        self.update()
        super().changeEvent(e)


class SectionHeader(QWidget):
    """[01] TITLE ///////  ──────────── ┐"""

    def __init__(self, index: int, title: str):
        super().__init__()
        self.index, self.title = index, title
        self.setFixedHeight(32)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(mono(9.5, bold=True, spacing=2.0))
        y = 22
        p.setPen(ACCENT)
        idx = f"[{self.index:02d}]"
        p.drawText(0, y, idx)
        x = p.fontMetrics().horizontalAdvance(idx) + 12
        # The title is sprayed on, a little crooked.
        x += spray_text(p, QPointF(x, y + 1), self.title.lower(), graffiti(15), QColor(TEXT), -2, glow=24) + 14
        p.setPen(QPen(QColor(FAINT), 1))
        for i in range(8):
            p.drawLine(QPointF(x + i * 5, y), QPointF(x + i * 5 + 5, y - 9))
        x += 8 * 5 + 16
        right = self.width() - 1
        p.setPen(QPen(QColor(LINE_STRONG), 1))
        p.drawLine(QPointF(x, y - 4), QPointF(right - 16, y - 4))
        p.setPen(QPen(ACCENT, 1.5))
        p.drawLine(QPointF(right - 10, y - 4), QPointF(right, y - 4))
        p.drawLine(QPointF(right, y - 4), QPointF(right, y + 5))


class Segmented(QWidget):
    """Segmented control with a sliding selection pill."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]]):
        super().__init__()
        self.options = options
        self.index = 0
        self.pill = Spring(0, SNAPPY, self.update)
        self.hovers = [Spring(0, SNAPPY, self.update) for _ in options]
        self.setFixedHeight(40)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_value(self, key: str, animate: bool = True) -> None:
        keys = [k for k, _ in self.options]
        if key not in keys:
            return
        self.index = keys.index(key)
        (self.pill.to if animate and self.isVisible() else self.pill.snap)(self.index)
        self.update()

    def _seg_at(self, x: float) -> int:
        return min(max(int(x / (self.width() / len(self.options))), 0), len(self.options) - 1)

    def mouseMoveEvent(self, e):
        i = self._seg_at(e.position().x())
        for j, h in enumerate(self.hovers):
            h.to(1 if j == i else 0)

    def leaveEvent(self, e):
        for h in self.hovers:
            h.to(0)

    def mousePressEvent(self, e):
        if not self.isEnabled():
            return
        i = self._seg_at(e.position().x())
        if i != self.index:
            self.set_value(self.options[i][0])
            self.changed.emit(self.options[i][0])

    def changeEvent(self, e):
        self.update()
        super().changeEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self.options)
        gap = 6
        seg_w = (self.width() - gap * (n - 1)) / n
        enabled = self.isEnabled()
        for i, (_, label) in enumerate(self.options):
            r = QRectF(i * (seg_w + gap) + 0.5, 0.5, seg_w - 1, self.height() - 1)
            h = self.hovers[i].value
            p.setPen(QPen(blend(QColor(LINE_STRONG), with_alpha(ACCENT, 120), h * 0.8), 1))
            p.setBrush(with_alpha(ACCENT, int(6 * h)))
            p.drawRect(r)
        # Sliding pill.
        x = self.pill.value * (seg_w + gap)
        pill = QRectF(x + 0.5, 0.5, seg_w - 1, self.height() - 1)
        p.setPen(QPen(ACCENT if enabled else QColor(FAINT), 1))
        p.setBrush(with_alpha(ACCENT, 26 if enabled else 6))
        p.drawRect(pill)
        p.setFont(mono(9, spacing=1.8))
        for i, (_, label) in enumerate(self.options):
            r = QRectF(i * (seg_w + gap), 0, seg_w, self.height())
            closeness = max(0.0, 1 - abs(self.pill.value - i))
            color = blend(QColor(TEXT), ACCENT, closeness) if enabled else QColor(FAINT)
            p.setPen(color)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, label)


class _Labeled(Animated):
    def __init__(self, text: str, suffix: str = "", check_preset=SNAPPY):
        super().__init__(check_preset)
        self.setCheckable(True)
        self.setText(text)
        self.suffix = suffix
        self.setFont(mono(9, spacing=1.6))

    def sizeHint(self):
        return QSize(240, 28)

    def _label(self, p: QPainter, x: float):
        p.setFont(self.font())
        enabled = self.isEnabled()
        p.setPen(blend(QColor(TEXT), QColor("#ffffff"), self.hover.value * 0.6) if enabled else QColor(FAINT))
        y = self.height() / 2 + 5
        p.drawText(QPointF(x, y), self.text())
        if self.suffix:
            x += p.fontMetrics().horizontalAdvance(self.text()) + 12
            p.setPen(QColor(DIM) if enabled else QColor(FAINT))
            p.drawText(QPointF(x, y), self.suffix)


class Toggle(_Labeled):
    """Pill switch; the knob springs across with a little overshoot."""

    def __init__(self, text: str, suffix: str = ""):
        super().__init__(text, suffix, check_preset=BOUNCY)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = self.check.value
        enabled = self.isEnabled()
        r = QRectF(0.5, self.height() / 2 - 9, 40, 18)
        accent = ACCENT if enabled else QColor(FAINT)
        p.setPen(QPen(blend(QColor(LINE_STRONG), accent, min(c + self.hover.value * 0.3, 1)), 1))
        p.setBrush(blend(QColor(0, 0, 0, 60), with_alpha(accent, 230), c))
        p.drawRoundedRect(r, 9, 9)
        knob_x = mix(r.left() + 9, r.right() - 9, c)
        squash = 1 + 0.18 * self.press.value  # knob widens while held
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(blend(QColor(DIM), INK, c))
        p.drawEllipse(QPointF(knob_x, r.center().y()), 5.5 * squash, 5.5)
        self._label(p, 54)


class TickBox(_Labeled):
    """Square checkbox whose check mark draws itself on."""

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = min(max(self.check.value, 0.0), 1.0)
        enabled = self.isEnabled()
        accent = ACCENT if enabled else QColor(FAINT)
        r = QRectF(0.5, self.height() / 2 - 8, 16, 16)
        p.setPen(QPen(blend(QColor(LINE_STRONG), accent, max(c, self.hover.value * 0.5)), 1))
        p.setBrush(blend(QColor(0, 0, 0, 60), with_alpha(accent, 220), c))
        p.drawRect(r)
        if c > 0.01:
            a = QPointF(r.left() + 3.5, r.center().y())
            b = QPointF(r.left() + 7, r.bottom() - 4)
            d = QPointF(r.right() - 3, r.top() + 4)
            path = QPainterPath(a)
            if c < 0.4:
                path.lineTo(a + (b - a) * (c / 0.4))
            else:
                path.lineTo(b)
                path.lineTo(b + (d - b) * ((c - 0.4) / 0.6))
            p.setPen(QPen(INK, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        self._label(p, 30)


class ChoiceRow(Animated):
    """Selectable row (effects): an accent bar grows in from the left when chosen."""

    def __init__(self, index: int, text: str, note: str = ""):
        super().__init__()
        self.setCheckable(True)
        self.index, self.label, self.note = index, text, note
        self.setFixedHeight(46)
        self.setFont(mono(9, spacing=2.0))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, h = self.check.value, self.hover.value
        enabled = self.isEnabled()
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        r.translate(2 * self.press.value, 0)
        p.setPen(QPen(blend(blend(QColor(LINE_STRONG), with_alpha(ACCENT, 110), h), ACCENT, c), 1))
        p.setBrush(with_alpha(ACCENT, int(8 * h + 22 * c)))
        p.drawRect(r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(ACCENT)
        p.drawRect(QRectF(r.left(), r.top() + 6, 3 * c, r.height() - 12))
        p.setFont(self.font())
        text_color = blend(QColor(TEXT), ACCENT, c) if enabled else QColor(FAINT)
        p.setPen(blend(QColor(DIM), ACCENT, c) if enabled else QColor(FAINT))
        y = r.center().y() + 5
        p.drawText(QPointF(r.left() + 18, y), str(self.index))
        p.setPen(text_color)
        p.drawText(QPointF(r.left() + 52, y), self.label)
        if self.note:
            p.setFont(mono(8, spacing=1.0))
            p.setPen(QColor(DIM) if enabled else QColor(FAINT))
            w = p.fontMetrics().horizontalAdvance(self.note)
            p.drawText(QPointF(r.right() - w - 16, y), self.note)


class DropZone(Animated):
    """Dashed target; the dashes march while hovered or while files are dragged over."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(88)
        self._drag = Spring(0, SNAPPY, self.update)
        self._march = QTimer(self, interval=40, timeout=self.update)
        self.hover.on_change = self._on_hot_change
        self._drag.on_change = self._on_hot_change

    @property
    def hot(self) -> float:
        return max(self.hover.value, self._drag.value)

    def set_drag_active(self, on: bool) -> None:
        self._drag.to(1 if on else 0)

    def _on_hot_change(self) -> None:
        if self.hot > 0.05 and not motion.reduced_motion():
            self._march.start()
        else:
            self._march.stop()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        hot = self.hot
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        pen = QPen(with_alpha(ACCENT, int(mix(110, 230, hot))), 1)
        pen.setDashPattern([4, 3])
        pen.setDashOffset(-(time.perf_counter() * 14) % 7 if hot > 0.05 else 0)
        p.setPen(pen)
        p.setBrush(with_alpha(ACCENT, int(mix(6, 26, hot) + 20 * self._drag.value)))
        p.drawRect(r)

        lift = -2 * hot
        g = QRectF(22, r.center().y() - 13 + lift, 30, 26)
        p.setPen(QPen(blend(QColor(TEXT), ACCENT, hot), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(g)
        mountain = QPainterPath(QPointF(g.left() + 3, g.bottom() - 4))
        mountain.lineTo(g.left() + 11, g.top() + 12)
        mountain.lineTo(g.left() + 17, g.top() + 18)
        mountain.lineTo(g.left() + 21, g.top() + 14)
        mountain.lineTo(g.right() - 3, g.bottom() - 4)
        p.drawPath(mountain)
        p.drawEllipse(QPointF(g.right() - 8, g.top() + 7), 2.5, 2.5)

        title = "RELEASE TO ADD" if self._drag.value > 0.5 else "SELECT PHOTOS / VIDEOS"
        p.setFont(mono(9.5, spacing=1.4))
        p.setPen(QColor(TEXT))
        p.drawText(QPointF(72, r.center().y() - 4), title)
        p.setFont(mono(8, spacing=0.6))
        p.setPen(QColor(DIM))
        p.drawText(QPointF(72, r.center().y() + 16), "drop files here, or click   ctrl+O")

        cx, cy = r.right() - 24 + 4 * hot, r.center().y()
        p.setPen(QPen(blend(QColor(TEXT), ACCENT, hot), 1.4))
        p.drawLine(QPointF(cx - 4, cy - 7), QPointF(cx + 3, cy))
        p.drawLine(QPointF(cx + 3, cy), QPointF(cx - 4, cy + 7))


def describe_file(path: Path) -> str:
    """'5760x3240 · 4.2 MB · JPG' (or with duration for videos)."""
    from n0face.media import media_kind

    size_mb = path.stat().st_size / 1e6
    ext = path.suffix.lstrip(".").upper()
    try:
        if media_kind(path) == "image":
            from PIL import Image

            with Image.open(path) as im:
                dims = f"{im.width}x{im.height}"
        else:
            import cv2

            cap = cv2.VideoCapture(str(path))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            dims = f"{w}x{h}" + (f" · {n / fps:.0f}s" if fps else "")
    except Exception:  # noqa: BLE001 - metadata is cosmetic; never block on it
        dims = "?"
    return f"{dims}  ·  {size_mb:.1f} MB  ·  {ext}"


class FileRow(Animated):
    """File card: slides in on arrival, highlights when selected, has a remove (x) zone."""

    removed = Signal(object)
    selected = Signal(object)

    def __init__(self, path: Path, kind: str, delay_ms: int = 0):
        super().__init__()
        self.path, self.kind = path, kind
        self.meta = describe_file(path)
        self.setCheckable(True)
        self.setFixedHeight(54)
        self.setMouseTracking(True)
        self.appear = Spring(0, SMOOTH, self.update)
        self.x_hover = Spring(0, SNAPPY, self.update)
        self.clicked.connect(self._on_click)
        QTimer.singleShot(delay_ms, lambda: self.appear.to(1))

    def _x_rect(self) -> QRectF:
        return QRectF(self.width() - 40, 0, 40, self.height())

    def mouseMoveEvent(self, e):
        self.x_hover.to(1 if self._x_rect().contains(e.position()) else 0)
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        self.x_hover.to(0)
        super().leaveEvent(e)

    def _on_click(self) -> None:
        if self.x_hover.target > 0.5:
            self.removed.emit(self.path)
        else:
            self.setChecked(True)
            self.selected.emit(self.path)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        a = min(max(self.appear.value, 0.0), 1.0)
        p.setOpacity(a)
        c, h = self.check.value, self.hover.value
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        r.translate(18 * (1 - a), 0)
        p.setPen(QPen(blend(blend(QColor(LINE_STRONG), with_alpha(ACCENT, 90), h), with_alpha(ACCENT, 170), c), 1))
        p.setBrush(blend(QColor(0, 0, 0, 50), with_alpha(ACCENT, 20), c * 0.8))
        p.drawRect(r)
        p.setPen(QPen(blend(QColor(DIM), ACCENT, c), 2))
        p.drawLine(QPointF(r.left() + 1, r.top() + 8), QPointF(r.left() + 1, r.bottom() - 8))

        # Kind glyph.
        gx, gy = r.left() + 22, r.center().y()
        p.setPen(QPen(blend(QColor(DIM), ACCENT, c), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.kind == "video":
            tri = QPainterPath(QPointF(gx - 4, gy - 6))
            tri.lineTo(gx + 6, gy)
            tri.lineTo(gx - 4, gy + 6)
            tri.closeSubpath()
            p.drawPath(tri)
        else:
            p.drawRect(QRectF(gx - 6, gy - 5, 12, 10))
            p.drawLine(QPointF(gx - 4, gy + 3), QPointF(gx, gy - 1))
            p.drawLine(QPointF(gx, gy - 1), QPointF(gx + 4, gy + 3))

        fm_name = mono(9, spacing=0.6)
        p.setFont(fm_name)
        avail = r.width() - 44 - 48
        name = QFontMetricsF(fm_name).elidedText(self.path.name, Qt.TextElideMode.ElideMiddle, avail)
        p.setPen(QColor(TEXT))
        p.drawText(QPointF(r.left() + 44, r.center().y() - 3), name)
        p.setFont(mono(7.5, spacing=0.6))
        p.setPen(QColor(DIM))
        p.drawText(QPointF(r.left() + 44, r.center().y() + 14), self.meta)

        # Remove (x), revealed more strongly on hover.
        xr = self._x_rect().translated(18 * (1 - a), 0)
        xh = self.x_hover.value
        cx, cy = xr.center().x(), xr.center().y()
        s = 4.5
        p.setPen(QPen(blend(with_alpha(QColor(DIM), int(120 + 100 * h)), QColor(LIVE), xh), 1.4))
        p.drawLine(QPointF(cx - s, cy - s), QPointF(cx + s, cy + s))
        p.drawLine(QPointF(cx + s, cy - s), QPointF(cx - s, cy + s))


class Callout(QWidget):
    """Bordered note with an icon; text and color crossfade when the message changes."""

    def __init__(self, text: str = "", kind: str = "info"):
        super().__init__()
        self.text, self.kind = text, kind
        self._next: tuple[str, str] | None = None
        self.fade = Spring(1, SNAPPY, self._on_fade)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.font_ = mono(8.5, spacing=0.8)

    def set(self, text: str, kind: str) -> None:
        if (text, kind) == (self.text, self.kind):
            return
        if not self.isVisible() or motion.reduced_motion():
            self.text, self.kind = text, kind
            self.updateGeometry()
            self.update()
            return
        self._next = (text, kind)
        self.fade.to(0)

    def _on_fade(self) -> None:
        if self._next is not None and self.fade.value < 0.08:
            self.text, self.kind = self._next
            self._next = None
            self.updateGeometry()
            self.fade.to(1)
        self.update()

    def _text_rect(self, width: float) -> QRectF:
        return QRectF(48, 14, max(width - 48 - 16, 10), 1000)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        br = QFontMetricsF(self.font_).boundingRect(self._text_rect(w), int(Qt.TextFlag.TextWordWrap), self.text)
        return int(br.height() + 28)

    def sizeHint(self):
        return QSize(300, self.heightForWidth(320))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(WARN) if self.kind == "warn" else QColor(ACCENT)
        f = min(max(self.fade.value, 0.0), 1.0)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(with_alpha(color, int(140 * (0.5 + 0.5 * f))), 1))
        p.setBrush(with_alpha(color, 14))
        p.drawRect(r)
        p.setOpacity(f)
        cx, cy = 24, 25
        p.setPen(QPen(color, 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.kind == "warn":
            tri = QPainterPath(QPointF(cx, cy - 8))
            tri.lineTo(cx + 9, cy + 7)
            tri.lineTo(cx - 9, cy + 7)
            tri.closeSubpath()
            p.drawPath(tri)
            p.drawLine(QPointF(cx, cy - 2), QPointF(cx, cy + 2))
            p.drawPoint(QPointF(cx, cy + 4.5))
        else:
            p.drawEllipse(QPointF(cx, cy), 8, 8)
            p.drawLine(QPointF(cx, cy - 1), QPointF(cx, cy + 4))
            p.drawPoint(QPointF(cx, cy - 4))
        p.setFont(self.font_)
        p.setPen(QColor(WARN) if self.kind == "warn" else QColor(DIM))
        p.drawText(self._text_rect(self.width()), int(Qt.TextFlag.TextWordWrap), self.text)


class StatusRow(QWidget):
    """Status line; a live status breathes instead of hard-blinking."""

    def __init__(self, label: str):
        super().__init__()
        self.setFixedHeight(32)
        self.label = label
        self.value = ""
        self.color = QColor(DIM)
        self.live = False
        self.flash = Spring(0, SMOOTH, self.update)
        self._breath = QTimer(self, interval=50, timeout=self.update)

    def set(self, value: str, color: str, live: bool = False) -> None:
        if value != self.value:
            self.flash.snap(1)
            self.flash.to(0)
        self.value, self.color, self.live = value, QColor(color), live
        if live and not motion.reduced_motion():
            self._breath.start()
        else:
            self._breath.stop()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height() / 2
        breath = 0.55 + 0.45 * math.sin(time.perf_counter() * 4.2) if self.live and self._breath.isActive() else 1.0
        p.setPen(Qt.PenStyle.NoPen)
        if self.live:
            p.setBrush(with_alpha(self.color, int(60 * breath)))
            p.drawEllipse(QPointF(6, y), 4 + 4 * breath, 4 + 4 * breath)
        p.setBrush(with_alpha(self.color, int(255 * (0.5 + 0.5 * breath))))
        p.drawEllipse(QPointF(6, y), 4, 4)
        p.setFont(mono(9, spacing=1.8))
        p.setPen(QColor(DIM))
        p.drawText(QPointF(24, y + 5), self.label)
        p.setPen(blend(self.color, QColor("#ffffff"), self.flash.value * 0.7))
        w = p.fontMetrics().horizontalAdvance(self.value)
        p.drawText(QPointF(self.width() - w - 2, y + 5), self.value)


class ActionButton(Animated):
    """[  PROCESS 1 FILE  ->  with corner brackets.

    While running it turns red, fills left-to-right with progress and shimmers; a success
    flash plays when work completes.
    """

    def __init__(self):
        super().__init__()
        self.setFixedHeight(60)
        self.running = False
        self.progress = Spring(0, SMOOTH, self.update)
        self.flash = Spring(0, SMOOTH, self.update)
        self.state = Spring(0, SNAPPY, self.update)  # 0 idle (accent) -> 1 running (red)
        self._shimmer = QTimer(self, interval=33, timeout=self.update)

    def set_running(self, running: bool) -> None:
        self.running = running
        self.state.to(1 if running else 0)
        if running:
            self.progress.snap(0)
            if not motion.reduced_motion():
                self._shimmer.start()
        else:
            self._shimmer.stop()
            self.progress.to(0)

    def set_progress(self, frac: float) -> None:
        self.progress.to(min(max(frac, 0.0), 1.0))

    def celebrate(self) -> None:
        self.flash.snap(1)
        self.flash.to(0)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled = self.isEnabled()
        color = blend(QColor(ACCENT), QColor(LIVE), self.state.value) if enabled else QColor(FAINT)
        h = self.hover.value if enabled else 0
        scale = 1 - 0.025 * self.press.value
        full = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        r = QRectF(0, 0, full.width() * scale, full.height() * scale)
        r.moveCenter(full.center())

        p.setPen(QPen(with_alpha(color, int(150 + 60 * h)), 1))
        p.setBrush(with_alpha(color, int(14 + 26 * h + 120 * self.flash.value)))
        p.drawRect(r)
        if self.running or self.progress.value > 0.002:
            fill = QRectF(r.left(), r.top(), r.width() * self.progress.value, r.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(with_alpha(color, 46))
            p.drawRect(fill)
        if self._shimmer.isActive():
            x = r.left() + ((time.perf_counter() * 0.6) % 1.4 - 0.2) * r.width()
            p.setPen(QPen(with_alpha(color, 90), 2))
            p.drawLine(QPointF(x, r.top() + 4), QPointF(x - 10, r.bottom() - 4))

        p.setPen(QPen(color, 2))
        brackets(p, r, 12 + 4 * h)
        font = graffiti(17)
        p.setFont(font)
        label = self.text().lower()
        w = p.fontMetrics().horizontalAdvance(label)
        spray_text(p, QPointF(r.center().x() - w / 2, r.center().y() + 8), label, font, color, -2, glow=40)
        p.setFont(mono(11))
        p.drawText(QRectF(r.left() + 18, r.top(), 20, r.height()), Qt.AlignmentFlag.AlignVCenter, "[")
        cx, cy = r.right() - 26 + 5 * h, r.center().y()
        p.setPen(QPen(color, 1.5))
        p.drawLine(QPointF(cx - 7, cy), QPointF(cx + 6, cy))
        p.drawLine(QPointF(cx + 6, cy), QPointF(cx + 1, cy - 5))
        p.drawLine(QPointF(cx + 6, cy), QPointF(cx + 1, cy + 5))


class TabBar(QWidget):
    """Tabs with a spring-driven underline that slides between them."""

    changed = Signal(int)

    def __init__(self, labels: list[str]):
        super().__init__()
        self.labels = labels
        self.current = 0
        self.underline = Spring(0, SNAPPY, self.update)
        self.stretch = Spring(0, SMOOTH, self.update)  # underline widens mid-flight
        self.hovers = [Spring(0, SNAPPY, self.update) for _ in labels]
        self.setFixedHeight(52)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_current(self, i: int) -> None:
        if i == self.current:
            return
        distance = abs(i - self.current)
        self.current = i
        self.underline.to(i)
        self.stretch.snap(min(distance, 2) * 0.35)
        self.stretch.to(0)
        self.changed.emit(i)

    def _tab_at(self, x: float) -> int:
        return min(max(int(x / (self.width() / len(self.labels))), 0), len(self.labels) - 1)

    def mouseMoveEvent(self, e):
        i = self._tab_at(e.position().x())
        for j, h in enumerate(self.hovers):
            h.to(1 if j == i else 0)

    def leaveEvent(self, e):
        for h in self.hovers:
            h.to(0)

    def mousePressEvent(self, e):
        self.set_current(self._tab_at(e.position().x()))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width() / len(self.labels)
        p.setPen(QPen(QColor(LINE_STRONG), 1))
        p.drawLine(QPointF(0, self.height() - 1), QPointF(self.width(), self.height() - 1))
        for i, label in enumerate(self.labels):
            r = QRectF(i * w, 0, w, self.height())
            near = max(0.0, 1 - abs(self.underline.value - i))
            h = self.hovers[i].value
            p.setFont(mono(7, spacing=1.2))
            p.setPen(blend(QColor(FAINT), ACCENT, near))
            p.drawText(r.adjusted(0, 9, 0, -24), Qt.AlignmentFlag.AlignCenter, f"{i + 1:02d}")
            p.setFont(mono(8.5, bold=near > 0.5, spacing=2.0))
            p.setPen(blend(blend(QColor(DIM), QColor(TEXT), h * 0.7), QColor(TEXT), near))
            p.drawText(r.adjusted(0, 20, 0, -4), Qt.AlignmentFlag.AlignCenter, label)
        x = self.underline.value * w
        grow = self.stretch.value * w
        p.setPen(QPen(ACCENT, 2))
        p.drawLine(QPointF(x + 16 - grow / 2, self.height() - 1.5), QPointF(x + w - 16 + grow / 2, self.height() - 1.5))


class Swatch(Animated):
    def __init__(self, color: QColor, name: str):
        super().__init__()
        self.color = color
        self.setCheckable(True)
        self.setToolTip(name)
        self.setFixedSize(48, 32)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, h = self.check.value, self.hover.value
        inset = 5 - 1.5 * h + 1.0 * self.press.value
        r = QRectF(self.rect()).adjusted(inset, inset, -inset, -inset)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.color)
        p.drawRect(r)
        if c > 0.01:
            ring = QRectF(self.rect()).adjusted(1, 1, -1, -1)
            p.setPen(QPen(with_alpha(QColor(TEXT), int(255 * min(c, 1))), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            grow = 4 * (1 - min(c, 1))
            p.drawRect(ring.adjusted(grow, grow, -grow, -grow))


class AccessoryTile(Animated):
    """A square picker tile: the accessory's icon over its name, lit when chosen."""

    def __init__(self, kind: str, icon_for):
        super().__init__()
        self.kind = kind
        self._icon_for = icon_for  # (kind, px, QColor) -> QImage, re-asked when the accent changes
        self._cache: tuple | None = None
        self.setCheckable(True)
        self.setFixedSize(108, 92)
        self.setToolTip(kind.upper())

    def _icon(self, px: int):
        key = (px, ACCENT.rgb())
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self._icon_for(self.kind, px, QColor(ACCENT)))
        return self._cache[1]

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, h = self.check.value, self.hover.value
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(blend(blend(QColor(LINE_STRONG), with_alpha(ACCENT, 110), h), ACCENT, c), 1 + c))
        p.setBrush(with_alpha(ACCENT, int(6 * h + 20 * c)))
        p.drawRect(r)
        lift = 2 * self.press.value
        px = 52
        p.setOpacity(0.55 + 0.45 * max(c, h))
        p.drawImage(QPointF(r.center().x() - px / 2, 8 + lift), self._icon(px))
        p.setOpacity(1)
        p.setFont(mono(7.5, spacing=1.6))
        p.setPen(blend(QColor(DIM), ACCENT, c))
        p.drawText(QRectF(r.left(), r.bottom() - 26, r.width(), 20), Qt.AlignmentFlag.AlignCenter, self.kind.upper())


class PreviewTile(QWidget):
    """A picture with a caption under it (EXTRAS previews). Not clickable."""

    def __init__(self, path, caption: str):
        from PySide6.QtGui import QImage

        super().__init__()
        self.image = QImage(str(path))
        self.caption = caption
        self.setFixedSize(108, 120)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(LINE_STRONG), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        if not self.image.isNull():
            p.drawImage(QRectF(r.left() + 6, r.top() + 4, r.width() - 12, r.width() - 12), self.image)
        p.setFont(mono(7.5, spacing=1.6))
        p.setPen(QColor(DIM))
        p.drawText(QRectF(r.left(), r.bottom() - 20, r.width(), 16), Qt.AlignmentFlag.AlignCenter, self.caption)


class Slider(QWidget):
    """A thin track with an accent fill and a knob that springs to where you drag it."""

    changed = Signal(float)

    def __init__(self, lo: float, hi: float, value: float, fmt=lambda v: f"{v:.1f}"):
        super().__init__()
        self.lo, self.hi, self.fmt = lo, hi, fmt
        self.value = value
        self.knob = Spring(self._frac(value), SNAPPY, self.update)
        self.hover = Spring(0, SNAPPY, self.update)
        self.setFixedHeight(28)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _frac(self, v: float) -> float:
        return (v - self.lo) / (self.hi - self.lo)

    def set_value(self, v: float, emit: bool = False) -> None:
        v = min(max(v, self.lo), self.hi)
        self.value = v
        self.knob.to(self._frac(v)) if self.isVisible() else self.knob.snap(self._frac(v))
        if emit:
            self.changed.emit(v)
        self.update()

    def _track(self) -> tuple[float, float]:
        return 8.0, self.width() - 56.0

    def _from_x(self, x: float) -> None:
        a, b = self._track()
        self.set_value(self.lo + (self.hi - self.lo) * min(max((x - a) / (b - a), 0), 1), emit=True)

    def mousePressEvent(self, e):
        self._from_x(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._from_x(e.position().x())

    def mouseDoubleClickEvent(self, e):  # double-click resets to the middle / default
        self.set_value(self.lo + (self.hi - self.lo) / 2 if self.lo < 0 else 1.0, emit=True)

    def enterEvent(self, e):
        self.hover.to(1)

    def leaveEvent(self, e):
        self.hover.to(0)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        a, b = self._track()
        y = self.height() / 2
        p.setPen(QPen(QColor(LINE_STRONG), 2))
        p.drawLine(QPointF(a, y), QPointF(b, y))
        x = a + (b - a) * self.knob.value
        p.setPen(QPen(ACCENT, 2))
        p.drawLine(QPointF(a, y), QPointF(x, y))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(ACCENT)
        r = 5 + 1.5 * self.hover.value
        p.drawEllipse(QPointF(x, y), r, r)
        p.setFont(mono(8, spacing=1.0))
        p.setPen(QColor(TEXT))
        p.drawText(QRectF(b + 8, 0, 48, self.height()), Qt.AlignmentFlag.AlignVCenter, self.fmt(self.value))


class AnimatedStack(QStackedWidget):
    """Stacked pages that fade and slide in when switched."""

    def __init__(self):
        super().__init__()
        self.reveal = Spring(1, SMOOTH, self._step)
        self._effect: QGraphicsOpacityEffect | None = None
        self._page: QWidget | None = None
        self._direction = 1

    def slide_to(self, index: int) -> None:
        if index == self.currentIndex():
            return
        self._direction = 1 if index > self.currentIndex() else -1
        self._finish()
        self.setCurrentIndex(index)
        if motion.reduced_motion() or not self.isVisible():
            return
        self.reveal.snap(0)  # before _page is set, so this snap doesn't end the transition
        self._page = self.currentWidget()
        self._effect = QGraphicsOpacityEffect(self._page)
        self._effect.setOpacity(0)
        self._page.setGraphicsEffect(self._effect)
        self.reveal.to(1)

    def _step(self) -> None:
        if self._page is None:
            return
        v = min(max(self.reveal.value, 0.0), 1.0)
        self._effect.setOpacity(v)
        self._page.move(int(24 * (1 - v) * self._direction), 0)
        if not self.reveal.moving:
            self._finish()

    def _finish(self) -> None:
        if self._page is not None:
            self._page.setGraphicsEffect(None)  # drop the offscreen render cost once settled
            self._page.move(0, 0)
        self._page = self._effect = None
