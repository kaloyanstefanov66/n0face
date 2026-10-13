"""Main screen: the processed image/feed, or the interactive 3D mask preview.

Performance model: the 3D frames are rendered on RenderThread; static chrome (frame, ticks,
crosshairs, checklist, thumbnail) is cached in a pixmap per size/theme; a paint is a few blits
plus a little text. The animation tick only runs while something is actually moving.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from n0face import __version__
from n0face.app import motion
from n0face.app.motion import GENTLE, SMOOTH, SNAPPY, Spring, ease_out_cubic, mix
from n0face.app.preview3d import MaskScene, Pose, rotation
from n0face.app.services import RenderRequest, RenderThread
from n0face.app.style import DIM, FAINT, LINE_STRONG, LIVE, OK, TEXT, WARN, graffiti, mono, spray_text, with_alpha
from n0face.app.widgets import ACCENT, brackets

MAX_RENDER = 720  # longest side of the 3D render, in device pixels; upscaled smoothly
MAX_SPIN = 6.0  # rad/s fling cap
ANONYMITY = {"mask": "NONE", "obfuscate": "PARTIAL", "anonymize": "HIGH"}
SHORTCUTS = (
    ("CTRL+O", "add photos / videos"),
    ("CTRL+ENTER", "process files / engage camera"),
    ("ESC", "cancel / disengage / close this"),
    ("CTRL+1..4", "switch sidebar tab"),
    ("M", "next privacy mode"),
    ("1  2  3", "choose effect"),
    ("R", "reset 3D view"),
    ("F11", "maximize"),
    ("?", "toggle this help"),
)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Orbit:
    """3D view physics: drag with inertia, eased zoom, springy reset, blended auto-sway."""

    def __init__(self):
        self.yaw = self.pitch = 0.0
        self.vyaw = self.vpitch = 0.0
        self.zoom = Spring(1.0, SMOOTH)
        self.auto = Spring(1.0, GENTLE)  # weight of the automatic sway
        self.auto_enabled = True
        self.dragging = False
        self.returning = False
        self.idle = 0.0

    @staticmethod
    def sway(t: float) -> tuple[float, float]:
        return 0.75 * math.sin(t * 0.35), 0.12 * math.sin(t * 0.27)

    def pose(self, t: float) -> Pose:
        sy, sp = self.sway(t)
        w = min(max(self.auto.value, 0.0), 1.0)
        return Pose(mix(self.yaw, sy, w), mix(self.pitch, sp, w), self.zoom.value)

    def start_drag(self, t: float) -> None:
        p = self.pose(t)
        self.yaw, self.pitch = p.yaw, p.pitch
        self.auto.snap(0)
        self.vyaw = self.vpitch = 0.0
        self.dragging, self.returning = True, False

    def drag(self, dx: float, dy: float, dt: float) -> None:
        dyaw, dpitch = dx * 0.01, dy * 0.01
        self.yaw += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch, -1.2, 1.2))
        if dt > 0:  # smoothed release velocity, capped so a flick can't send it spinning wildly
            self.vyaw = float(np.clip(mix(self.vyaw, dyaw / dt, 0.5), -MAX_SPIN, MAX_SPIN))
            self.vpitch = float(np.clip(mix(self.vpitch, dpitch / dt, 0.5), -MAX_SPIN, MAX_SPIN))

    def end_drag(self, stale: bool) -> None:
        self.dragging = False
        self.idle = 0.0
        if stale or motion.reduced_motion():  # the pointer stopped before release: no fling
            self.vyaw = self.vpitch = 0.0

    def zoom_by(self, steps: float) -> None:
        self.zoom.to(float(np.clip(self.zoom.target * (1.15**steps), 0.5, 3.0)))

    def reset(self) -> None:
        self.zoom.to(1.0)
        self.vyaw = self.vpitch = 0.0
        self.returning, self.idle = True, 0.0

    def step(self, dt: float) -> None:
        if self.dragging:
            return
        self.yaw += self.vyaw * dt
        self.pitch = float(np.clip(self.pitch + self.vpitch * dt, -1.2, 1.2))
        friction = math.exp(-dt * 3.2)
        self.vyaw *= friction
        self.vpitch *= friction
        if self.returning:
            k = 1 - math.exp(-dt * 7)
            self.yaw = mix(wrap_angle(self.yaw), 0.0, k)
            self.pitch = mix(self.pitch, 0.0, k)
            if abs(self.yaw) < 1e-3 and abs(self.pitch) < 1e-3:
                self.returning = False
        if abs(self.vyaw) + abs(self.vpitch) < 0.02:
            self.idle += dt
        if self.auto_enabled and self.idle > 3.0 and self.auto.target < 1 and not motion.reduced_motion():
            self.yaw = wrap_angle(self.yaw)
            self.auto.to(1)  # hand back to the sway, blended so it never jumps

    def moving(self) -> bool:
        return (
            self.dragging or self.returning or abs(self.vyaw) + abs(self.vpitch) > 0.02
            or self.zoom.moving or self.auto.moving
        )


@dataclass
class Toast:
    title: str
    body: str
    kind: str  # "ok" | "warn" | "error" | "info"
    born: float
    ttl: float
    spring: Spring


class Viewport(QWidget):
    def __init__(self, state):
        """`state()` returns (mode, effect, theme, auto_rotate)."""
        super().__init__()
        self.state = state
        self.scene = MaskScene()
        self.renderer = RenderThread(self.scene)
        self.renderer.rendered.connect(self._on_rendered)
        self.renderer.start()
        self.orbit = Orbit()

        self.image: QImage | None = None
        self.prev_image: QImage | None = None
        self.frame_img: QImage | None = None
        self.frame_meta = None
        self.prev_frame: tuple[QImage, QRectF] | None = None
        self._in_flight = False
        self._render_rect = QRectF()
        self._render_scale = 1.0
        self.hover_pos: QPointF | None = None
        self.hover_info: dict | None = None

        self.busy_text = ""
        self.hint: str | list[str] = ""  # persistent status line (e.g. camera on, no face yet)
        self.caption = "OUTPUT // PROCESSED COPY PREVIEW"
        self.message = ("NO FILES", "drop photos / videos here, or press ctrl+O")
        self.cam_live = False
        self.resolution = "--"
        self.fps = "--"
        self.toasts: list[Toast] = []

        self.intro = Spring(0, GENTLE, self.update)
        self.reveal = Spring(1, SMOOTH, self.update)
        self.xfade = Spring(1, SMOOTH, self.update)
        self.hover_s = Spring(0, SNAPPY, self.update)
        self.help = Spring(0, SNAPPY, self.update)
        self._chrome: QPixmap | None = None
        self._chrome_key = None
        self._t0 = time.perf_counter()
        self._last_tick = self._t0
        self._drag_last: tuple[QPointF, float] | None = None
        self._frames = 0
        self._fps_t = self._t0

        self.setMouseTracking(True)
        self.setMinimumSize(560, 420)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._tick_timer = QTimer(self, timeout=self._tick)
        self._tick_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._tick_timer.start(16)

    # ---- lifecycle --------------------------------------------------------------------------

    def showEvent(self, e):
        if self.intro.value == 0:
            self.intro.to(1)
        super().showEvent(e)

    def shutdown(self) -> None:
        self._tick_timer.stop()
        self.renderer.stop()
        self.renderer.wait(2000)

    @property
    def showing_3d(self) -> bool:
        return self.image is None

    def now(self) -> float:
        return time.perf_counter() - self._t0

    # ---- public state -------------------------------------------------------------------------

    def set_image(self, img: QImage | None, reveal: bool = False) -> None:
        """Show a processed image/feed frame (None = back to the 3D preview)."""
        if reveal or (img is None) != (self.image is None):
            # The outgoing content (previous image, or the 3D preview) fades under the reveal.
            self.prev_image = self.image
            self.reveal.snap(0)
            self.reveal.to(1)
        self.image = img
        if img is not None:
            self.resolution = f"{img.width()}x{img.height()}"
            self.busy_text = ""
            if not self.cam_live:
                self.fps = "--"  # a still preview has no frame rate
        self.update()

    def style_changed(self) -> None:
        """Mode/effect/theme changed: cross-dissolve the 3D preview into the new style."""
        if self.showing_3d and self.frame_img is not None and not motion.reduced_motion():
            self.prev_frame = (self.frame_img, self._frame_rect())
            self.xfade.snap(0)
            self.xfade.to(1)
        self._chrome_key = None
        self.update()

    def toast(self, title: str, body: str = "", kind: str = "info", ttl: float = 4.0) -> None:
        s = Spring(0, SNAPPY, self.update)
        self.toasts.append(Toast(title, body, kind, self.now(), ttl, s))
        self.toasts = self.toasts[-3:]
        s.to(1)

    def toggle_help(self, on: bool | None = None) -> None:
        self.help.to((1 if self.help.target < 0.5 else 0) if on is None else (1 if on else 0))

    def reset_view(self) -> None:
        self.orbit.reset()

    def set_auto_rotate(self, on: bool) -> None:
        self.orbit.auto_enabled = on
        if not on:
            p = self.orbit.pose(self.now())
            self.orbit.yaw, self.orbit.pitch = p.yaw, p.pitch
            self.orbit.auto.to(0)
        else:
            self.orbit.idle = 3.1

    # ---- animation loop -----------------------------------------------------------------------

    def _tick(self) -> None:
        if not self.isVisible() or (self.window() and self.window().isMinimized()):
            return
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now
        self.orbit.step(dt)
        t = self.now()
        self._expire_toasts(t)

        if self.showing_3d:
            interactive = self.orbit.moving() or self.hover_pos is not None or self.xfade.moving or self.intro.moving
            animate = not motion.reduced_motion() or self.orbit.moving()
            # 60 fps while you interact or something transitions; the slow idle sway is just as
            # smooth at 30, for half the CPU.
            # In the background (another app focused) a gentle 10 fps is plenty.
            idle_fps = 30 if self.window().isActiveWindow() else 10
            due = interactive or now - getattr(self, "_last_request", 0.0) >= 1 / idle_fps
            if (animate or self.frame_img is None) and due and not self._in_flight:
                self._last_request = now
                self._request_frame(t)
        elif not motion.reduced_motion() and int(now * 20) != int((now - dt) * 20):
            self.update()  # slow chrome animation (globe, ruler markers) ~20 fps

    def _request_frame(self, t: float) -> None:
        mode, effect, theme, _ = self.state()
        box = self._scene_box()
        if box.width() < 8 or box.height() < 8:
            return
        scale = min(1.0, MAX_RENDER / max(box.width(), box.height())) * self.devicePixelRatioF()
        w, h = max(int(box.width() * scale), 2), max(int(box.height() * scale), 2)
        hover = None
        if self.hover_pos is not None and box.contains(self.hover_pos):
            hover = ((self.hover_pos.x() - box.left()) * scale, (self.hover_pos.y() - box.top()) * scale)
        anim_t = t if not motion.reduced_motion() else 1.0
        self._in_flight = True
        self._render_rect, self._render_scale = box, scale
        self.renderer.submit(RenderRequest(w, h, effect, mode, theme, anim_t, self.orbit.pose(t), hover, scale))

    def _on_rendered(self, img: QImage, meta) -> None:
        self._in_flight = False
        self.frame_img, self.frame_meta = img, meta
        info = meta.info
        if info is not None:
            self.hover_info = info
            self.hover_s.to(1)
        else:
            self.hover_s.to(0)
        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 0.5:
            if self.showing_3d:
                self.fps = f"{self._frames / (now - self._fps_t):.1f}"
            self._frames, self._fps_t = 0, now
        self.update()

    def _expire_toasts(self, t: float) -> None:
        for toast in self.toasts:
            if t - toast.born > toast.ttl and toast.spring.target > 0:
                toast.spring.to(0)
        self.toasts = [x for x in self.toasts if x.spring.target > 0 or x.spring.value > 0.01]

    # ---- interaction ------------------------------------------------------------------------

    def mousePressEvent(self, e):
        if self.showing_3d and e.button() == Qt.MouseButton.LeftButton:
            self.orbit.start_drag(self.now())
            self._drag_last = (e.position(), time.perf_counter())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.hover_pos = None

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag_last is not None:
            last, t_last = self._drag_last
            now = time.perf_counter()
            d = pos - last
            self.orbit.drag(d.x(), d.y(), now - t_last)
            self._drag_last = (pos, now)
        elif self.showing_3d:
            self.hover_pos = pos if self._scene_box().contains(pos) else None
            if self.hover_pos is None:
                self.hover_s.to(0)
            if motion.reduced_motion() and not self._in_flight:
                self._request_frame(self.now())

    def mouseReleaseEvent(self, e):
        if self._drag_last is not None:
            stale = time.perf_counter() - self._drag_last[1] > 0.08
            self.orbit.end_drag(stale)
            self._drag_last = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, e):
        self.hover_pos = None
        self.hover_s.to(0)

    def wheelEvent(self, e):
        if self.showing_3d:
            self.orbit.zoom_by(e.angleDelta().y() / 120)

    def mouseDoubleClickEvent(self, e):
        if self.showing_3d:
            self.orbit.reset()

    def resizeEvent(self, e):
        self._chrome_key = None
        super().resizeEvent(e)

    # ---- geometry -----------------------------------------------------------------------------

    def _frame_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(1, 30, -1, -1)

    def _inner(self) -> QRectF:
        return self._frame_rect().adjusted(18, 18, -18, -18)

    def _scene_box(self) -> QRectF:
        area = self._inner().adjusted(60, 50, -60, -112)
        side = min(area.width(), area.height())
        box = QRectF(0, 0, area.width(), side)
        box.moveCenter(area.center())
        return box

    def image_area(self) -> QSize:
        """Device-pixel size images are shown at (lets workers pre-scale camera frames)."""
        a = self._inner().adjusted(70, 120, -70, -100)
        dpr = self.devicePixelRatioF()
        return QSize(int(a.width() * dpr), int(a.height() * dpr))

    # ---- painting -----------------------------------------------------------------------------

    def paintEvent(self, _):
        t = self.now()
        mode, effect, _, _ = self.state()
        intro = ease_out_cubic(self.intro.value)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        frame, inner = self._frame_rect(), self._inner()

        p.setOpacity(intro)
        p.drawPixmap(0, 0, self._chrome_pixmap())
        p.setOpacity(1)
        self._brackets(p, frame, intro)

        if self.showing_3d or self.reveal.value < 1:
            self._draw_scene(p, inner, mode, intro, t)
        if self.image is not None or (self.prev_image is not None and self.reveal.value < 1):
            self._draw_images(p, inner)

        p.setOpacity(intro)
        self._readout_left(p, inner, t)
        self._readout_right(p, inner, mode, effect, t)
        self._ruler_markers(p, frame, t)
        p.setOpacity(1)
        if self.busy_text or self.hint:
            self._busy(p, inner, t, self.busy_text or self.hint)
        self._draw_toasts(p, inner)
        if self.help.value > 0.01:
            self._draw_help(p, inner)

    def _chrome_pixmap(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        key = (self.size().width(), self.size().height(), dpr, ACCENT.rgb(), self.caption)
        if self._chrome is not None and key == self._chrome_key:
            return self._chrome
        pm = QPixmap(int(self.width() * dpr), int(self.height() * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame, inner = self._frame_rect(), self._inner()
        self._caption(p)
        p.setBrush(QColor(4, 9, 12, 120))
        p.setPen(QPen(QColor(LINE_STRONG), 1))
        p.drawRect(frame)
        p.setPen(QPen(with_alpha(ACCENT, 40), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(frame.adjusted(10, 10, -10, -10))
        self._crosshairs(p, inner)
        self._ruler_ticks(p, frame)
        self._checklist(p, inner)
        self._signature(p, inner)
        p.end()
        self._chrome, self._chrome_key = pm, key
        return pm

    def _caption(self, p: QPainter):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(ACCENT, 200))
        p.drawRect(QRectF(2, 9, 11, 11))
        p.setPen(QPen(QColor("#061015"), 1.4))
        p.drawLine(QPointF(4, 11), QPointF(11, 18))
        p.drawLine(QPointF(11, 11), QPointF(4, 18))
        p.setFont(mono(9, spacing=2.0))
        p.setPen(QColor(TEXT))
        head, _, rest = self.caption.partition(" // ")
        p.drawText(24, 19, head)
        w = p.fontMetrics().horizontalAdvance(head + " ")
        p.setPen(QColor(DIM))
        p.drawText(24 + w, 19, f"// {rest}" if rest else "")

    def _brackets(self, p: QPainter, r: QRectF, intro: float):
        # On boot the corner brackets fly out from the center to the frame corners.
        c = r.center()
        rr = QRectF(
            QPointF(mix(c.x() - 40, r.left(), intro), mix(c.y() - 30, r.top(), intro)),
            QPointF(mix(c.x() + 40, r.right(), intro), mix(c.y() + 30, r.bottom(), intro)),
        )
        p.setPen(QPen(ACCENT, 2))
        brackets(p, rr, 26)

    def _crosshairs(self, p: QPainter, r: QRectF):
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        cols, rows = 6, 4
        for cx in range(1, cols):
            for cy in range(1, rows):
                x = r.left() + r.width() * cx / cols
                y = r.top() + r.height() * cy / rows
                p.drawLine(QPointF(x - 5, y), QPointF(x + 5, y))
                p.drawLine(QPointF(x, y - 5), QPointF(x, y + 5))
        c = r.center()
        p.setPen(QPen(QColor(255, 255, 255, 14), 1))
        p.drawLine(QPointF(c.x(), r.top() + 40), QPointF(c.x(), r.bottom() - 40))
        p.drawLine(QPointF(r.left() + 40, c.y()), QPointF(r.right() - 40, c.y()))

    def _ruler_ticks(self, p: QPainter, frame: QRectF):
        top, bottom = frame.top() + 70, frame.bottom() - 70
        for side, x in ((0, frame.left() + 22), (1, frame.right() - 22)):
            n = 34
            for i in range(n):
                y = top + (bottom - top) * i / (n - 1)
                long = i % 5 == 0
                p.setPen(QPen(QColor(DIM) if long else QColor(FAINT), 1))
                dx = (8 if long else 4) * (1 if side == 0 else -1)
                p.drawLine(QPointF(x, y), QPointF(x + dx, y))

    def _ruler_markers(self, p: QPainter, frame: QRectF, t: float):
        top, bottom = frame.top() + 70, frame.bottom() - 70
        tt = 0.0 if motion.reduced_motion() else t
        for side, x in ((0, frame.left() + 22), (1, frame.right() - 22)):
            marker = top + (bottom - top) * (0.5 + 0.35 * math.sin(tt * 0.4 + side * 2))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(OK) if side == 0 else ACCENT)
            p.drawRect(QRectF(x - 1 if side == 0 else x - 5, marker - 4, 6, 8))

    def _checklist(self, p: QPainter, r: QRectF):
        x, y = r.left() + 34, r.bottom() - 82
        p.setFont(mono(7.5, spacing=1.6))
        for k, label in enumerate(("LOCAL ONLY", "NO CLOUD", "NO TELEMETRY", "METADATA STRIPPED")):
            yy = y + k * 17
            p.setPen(QPen(with_alpha(ACCENT, 180), 1))
            p.setBrush(with_alpha(ACCENT, 150))
            p.drawRect(QRectF(x, yy - 8, 9, 9))
            p.setPen(QColor(DIM))
            p.drawText(int(x + 20), int(yy), label)

    def _signature(self, p: QPainter, r: QRectF):
        box = QRectF(r.right() - 92, r.bottom() - 96, 62, 70)
        p.setPen(QPen(QColor(LINE_STRONG), 1))
        p.setBrush(QColor(0, 0, 0, 60))
        p.drawRect(box)
        rot = rotation(0.35, 0.05)
        v = self.scene.v @ rot.T
        pts = np.stack([v[:, 0], -v[:, 1]], axis=1)
        pts = (pts - pts.min(axis=0)) / np.ptp(pts, axis=0).max()
        s = box.height() - 16
        ox = box.left() + (box.width() - s * 0.82) / 2
        oy = box.top() + 8
        p.setPen(QPen(with_alpha(ACCENT, 90), 0.6))
        for (x0, y0), (x1, y1) in pts[self.scene.edges[::2]]:
            p.drawLine(QPointF(ox + x0 * s, oy + y0 * s), QPointF(ox + x1 * s, oy + y1 * s))
        p.setFont(mono(7, spacing=1.2))
        p.setPen(QColor(DIM))
        p.drawText(QRectF(box.left() - 230, box.bottom() - 14, 220, 14), Qt.AlignmentFlag.AlignRight, "// YOUR FACE. YOUR DATA.")

    def _frame_target(self, img: QImage, meta) -> QRectF:
        box, s = self._render_rect, self._render_scale
        return QRectF(box.left() + meta.x / s, box.top() + meta.y / s, img.width() / s, img.height() / s)

    def _draw_scene(self, p: QPainter, r: QRectF, mode: str, intro: float, t: float):
        box = self._scene_box()
        out = 1 - ease_out_cubic(self.reveal.value) if self.image is not None else ease_out_cubic(self.reveal.value)
        base_opacity = intro * (out if self.image is not None or self.reveal.value < 1 else 1)

        x = ease_out_cubic(self.xfade.value)
        if self.prev_frame is not None and x < 1:
            img, rect = self.prev_frame
            p.setOpacity(base_opacity * (1 - x))
            # Glitch out: the old frame shears sideways in slices as it fades.
            slices = 6
            for i in range(slices):
                sr = QRectF(0, img.height() * i / slices, img.width(), img.height() / slices)
                dr = QRectF(rect.left(), rect.top() + rect.height() * i / slices, rect.width(), rect.height() / slices)
                dr.translate((1 - x) * 18 * math.sin(i * 2.3 + t * 30) * (1 if i % 2 else -1), 0)
                p.drawImage(dr, img, sr)
        if self.frame_img is not None and self.frame_meta is not None:
            p.setOpacity(base_opacity * (x if self.prev_frame is not None else 1))
            target = self._frame_target(self.frame_img, self.frame_meta)
            if intro < 1:  # boot: the model settles in from slightly smaller
                c = target.center()
                s = mix(0.9, 1.0, intro)
                target = QRectF(c.x() - target.width() * s / 2, c.y() - target.height() * s / 2, target.width() * s, target.height() * s)
            p.drawImage(target, self.frame_img)
        if x >= 1:
            self.prev_frame = None
        p.setOpacity(base_opacity)

        if self.image is None:
            if self.hover_s.value > 0.01 and self.hover_info is not None and self.frame_meta is not None:
                self._hover_label(p, box, self.hover_info, self.hover_s.value)
            if self.frame_meta is not None:
                self._gizmo(p, QPointF(box.left() + 30, box.bottom() - 30), self.frame_meta.rotation)
            self._scene_readout(p, r, box, mode)
            self._empty_message(p, r, box, t)
        p.setOpacity(1)

    def _scene_readout(self, p: QPainter, r: QRectF, box: QRectF, mode: str):
        pose = self.orbit.pose(self.now())
        p.setFont(mono(8, spacing=1.4))
        p.setPen(QColor(DIM))
        yaw = math.degrees(wrap_angle(pose.yaw))
        geometry = "SHELL   GEOMETRY HIDDEN" if mode == "anonymize" else "VERTS 468   TRIS 898"
        stats = f"YAW {yaw:+04.0f}°   PITCH {math.degrees(pose.pitch):+03.0f}°   ZOOM {pose.zoom:.2f}x   {geometry}"
        p.drawText(QRectF(r.left(), box.bottom() + 6, r.width(), 18), Qt.AlignmentFlag.AlignCenter, stats)
        p.setFont(mono(7.5, spacing=1.0))
        p.setPen(QColor(FAINT))
        p.drawText(
            QRectF(r.left(), box.bottom() + 24, r.width(), 18), Qt.AlignmentFlag.AlignCenter,
            "DRAG rotate · SCROLL zoom · DOUBLE-CLICK reset · HOVER inspect · ? keys",
        )

    def _empty_message(self, p: QPainter, r: QRectF, box: QRectF, t: float):
        title, hint = self.message
        if not title:
            return
        m = QRectF(0, 0, 380, 58)
        m.moveCenter(QPointF(r.center().x(), box.bottom() + 78))
        breathe = 0.5 + 0.5 * math.sin(t * 2.0) if not motion.reduced_motion() else 1.0
        p.setPen(QPen(with_alpha(ACCENT, int(140 + 115 * breathe)), 1.2))
        brackets(p, m, 9)
        font = graffiti(19)
        p.setFont(font)
        w = p.fontMetrics().horizontalAdvance(title.lower())
        spray_text(p, QPointF(m.center().x() - w / 2, m.top() + 26), title.lower(), font, ACCENT, -2)
        p.setFont(mono(8.5, spacing=1.0))
        p.setPen(QColor(DIM))
        p.drawText(m.adjusted(0, 30, 0, -4), Qt.AlignmentFlag.AlignCenter, hint)

    def _image_target(self, img: QImage, r: QRectF) -> QRectF:
        area = r.adjusted(70, 120, -70, -100)
        size = img.size().scaled(area.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRectF(0, 0, size.width(), size.height())
        target.moveCenter(area.center())
        return target

    def _draw_images(self, p: QPainter, r: QRectF):
        v = ease_out_cubic(self.reveal.value)
        if self.prev_image is not None and v < 1:
            p.setOpacity(1 - v)
            p.drawImage(self._image_target(self.prev_image, r), self.prev_image)
            p.setOpacity(1)
        if self.image is None:
            return
        target = self._image_target(self.image, r)
        if v < 1:
            # Scan-line reveal, top to bottom.
            cut = target.height() * v
            src = QRectF(0, 0, self.image.width(), self.image.height() * v)
            p.drawImage(QRectF(target.left(), target.top(), target.width(), cut), self.image, src)
            y = target.top() + cut
            p.setPen(QPen(with_alpha(ACCENT, 230), 2))
            p.drawLine(QPointF(target.left() - 12, y), QPointF(target.right() + 12, y))
            p.setPen(QPen(with_alpha(ACCENT, 60), 8))
            p.drawLine(QPointF(target.left(), y), QPointF(target.right(), y))
        else:
            p.drawImage(target, self.image)
            self.prev_image = None
        p.setPen(QPen(with_alpha(ACCENT, int(90 * v)), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(target.adjusted(-0.5, -0.5, 0.5, 0.5))

    def _hover_label(self, p: QPainter, box: QRectF, info: dict, k: float):
        s = self._render_scale
        if info.get("hidden"):
            if self.hover_pos is None:
                return
            pos = self.hover_pos
            lines = ["GEOMETRY HIDDEN", "anonymize renders no landmarks"]
        else:
            x, y = info["pos"]
            pos = QPointF(box.left() + x / s, box.top() + y / s)
            p.setPen(QPen(with_alpha(QColor(TEXT), int(255 * k)), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            rad = mix(14, 7, k)
            p.drawEllipse(pos, rad, rad)
            p.drawLine(pos + QPointF(-12, 0), pos + QPointF(-4, 0))
            p.drawLine(pos + QPointF(4, 0), pos + QPointF(12, 0))
            cx, cy, cz = info["cm"]
            lines = [f"LMK {info['index']:03d} · {info['region']}", f"x {cx:+.2f}  y {cy:+.2f}  z {cz:+.2f} cm"]
        anchor = pos + QPointF(46, -40)
        start = pos + QPointF(6, -6)
        p.setOpacity(k)
        p.setPen(QPen(with_alpha(ACCENT, 200), 1))
        p.drawLine(start, start + (anchor - start) * k)  # the leader line grows out
        p.setFont(mono(8, spacing=1.0))
        width = max(p.fontMetrics().horizontalAdvance(t) for t in lines) + 20
        rect = QRectF(anchor.x() + 8 * (1 - k), anchor.y() - 22, width, 42)
        p.setBrush(QColor(4, 10, 14, 220))
        p.drawRect(rect)
        p.setPen(QColor(TEXT))
        p.drawText(rect.adjusted(10, 4, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, lines[0])
        p.setPen(QColor(DIM))
        p.drawText(rect.adjusted(10, 21, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, lines[1])
        p.setOpacity(1)

    def _gizmo(self, p: QPainter, origin: QPointF, r: np.ndarray):
        p.setFont(mono(6.5))
        for axis, color, label in ((0, QColor(LIVE), "X"), (1, QColor(OK), "Y"), (2, QColor(ACCENT), "Z")):
            v = r[:, axis]
            end = origin + QPointF(v[0] * 20, -v[1] * 20)
            p.setPen(QPen(with_alpha(color, 200), 1.4))
            p.drawLine(origin, end)
            p.drawText(end + QPointF(2, 3), label)

    def _readout_left(self, p: QPainter, r: QRectF, t: float):
        x, y = r.left() + 34, r.top() + 34
        p.setFont(mono(8, spacing=1.6))
        p.setPen(QColor(DIM))
        p.drawText(int(x), int(y), "CAM")
        p.setPen(QColor(FAINT))
        p.drawText(int(x + 34), int(y), "|")
        bars_x = x + 48
        live = self.cam_live
        for i in range(18):
            on = live and (math.sin(t * 6 + i * 0.7) > -0.3)
            p.setPen(QPen(with_alpha(ACCENT, 230) if on else QColor(FAINT), 1.5))
            p.drawLine(QPointF(bars_x + i * 4, y - 9), QPointF(bars_x + i * 4, y))
        if live:
            pulse = 0.55 + 0.45 * math.sin(t * 4.2)
            p.setPen(with_alpha(QColor(LIVE), int(255 * pulse)))
            p.drawText(int(bars_x + 80), int(y), "● LIVE")
        rows = (("RES", self.resolution), ("FPS" if not self.showing_3d else "RENDER", self.fps))
        for k, (label, value) in enumerate(rows):
            p.setPen(QColor(DIM))
            p.drawText(int(x), int(y + 22 + k * 18), label)
            p.setPen(QColor(TEXT))
            p.drawText(int(x + 64), int(y + 22 + k * 18), value)

    def _readout_right(self, p: QPainter, r: QRectF, mode: str, effect: str, t: float):
        right = r.right() - 30
        y = r.top() + 34
        p.setFont(mono(8, spacing=1.4))
        rows = (
            (f"n0face v{__version__}", ""),
            ("PRIVACY", mode.upper()),
            ("EFFECT", effect.upper()),
            ("ANONYMITY", ANONYMITY.get(mode, "--")),
            ("PROCESSING", "LOCAL"),
        )
        gx = right - 34
        for k, (label, value) in enumerate(rows):
            yy = y + k * 17
            p.setPen(QColor(TEXT) if k == 0 else QColor(DIM))
            p.drawText(int(gx - 250), int(yy), label)
            if value:
                p.setPen(ACCENT if k != 3 or mode == "anonymize" else QColor(DIM))
                w = p.fontMetrics().horizontalAdvance(value)
                p.drawText(int(gx - 52 - w), int(yy), value)
        self._globe(p, QPointF(right - 6, y + 18), 22, 0.0 if motion.reduced_motion() else t)

    def _globe(self, p: QPainter, c: QPointF, rad: float, t: float):
        p.setPen(QPen(with_alpha(ACCENT, 170), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, rad, rad)
        for k in range(1, 4):
            w = abs(math.cos(t * 0.6 + k * math.pi / 4)) * rad
            p.drawEllipse(c, w, rad)
        for k in (-0.5, 0.0, 0.5):
            yy = c.y() + k * rad * 1.2
            half = math.sqrt(max(rad**2 - (yy - c.y()) ** 2, 0))
            p.drawLine(QPointF(c.x() - half, yy), QPointF(c.x() + half, yy))

    def _busy(self, p: QPainter, r: QRectF, t: float, text):
        """Status state: a sweeping bar plus copy that cycles while it works or waits."""
        lines = [text] if isinstance(text, str) else text
        text = lines[int(t / 1.1) % len(lines)]
        y = r.bottom() - 46
        w = 260
        bar = QRectF(r.center().x() - w / 2, y + 22, w, 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(LINE_STRONG))
        p.drawRect(bar)
        k = (t * 0.9) % 1.0
        seg = QRectF(bar.left() + (w - 70) * (0.5 - 0.5 * math.cos(k * 2 * math.pi)), bar.top() - 0.5, 70, 3)
        p.setBrush(ACCENT)
        p.drawRect(seg)
        p.setFont(mono(9, spacing=2.0))
        p.setPen(ACCENT)
        p.drawText(QRectF(r.left(), y, r.width(), 20), Qt.AlignmentFlag.AlignCenter, text)

    def _draw_toasts(self, p: QPainter, r: QRectF):
        colors = {"ok": QColor(OK), "warn": QColor(WARN), "error": QColor(LIVE), "info": QColor(ACCENT)}
        # The top-center strip between the two readout blocks is free in every state.
        y = r.top() + 16
        max_w = min(540.0, r.width() - 700)
        for toast in self.toasts:
            k = min(max(toast.spring.value, 0.0), 1.2)
            color = colors.get(toast.kind, QColor(ACCENT))
            p.setFont(mono(9, bold=True, spacing=2.0))
            w = max(p.fontMetrics().horizontalAdvance(toast.title), 0)
            p.setFont(mono(8, spacing=1.0))
            body = p.fontMetrics().elidedText(toast.body, Qt.TextElideMode.ElideRight, int(max_w - 52))
            w = min(max(w, p.fontMetrics().horizontalAdvance(body)) + 52, max_w)
            h = 52 if toast.body else 34
            rect = QRectF(r.center().x() - w / 2, y - 14 * (1 - k), w, h)
            p.setOpacity(min(k, 1.0))
            p.setPen(QPen(with_alpha(color, 170), 1))
            p.setBrush(QColor(4, 10, 14, 235))
            p.drawRect(rect)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawRect(QRectF(rect.left(), rect.top(), 3, rect.height()))
            p.setFont(mono(9, bold=True, spacing=2.0))
            p.setPen(color)
            p.drawText(QPointF(rect.left() + 22, rect.top() + 22), toast.title)
            if toast.body:
                p.setFont(mono(8, spacing=1.0))
                p.setPen(QColor(TEXT))
                p.drawText(QPointF(rect.left() + 22, rect.top() + 40), body)
            y += (h + 10) * min(k, 1.0)
        p.setOpacity(1)

    def _draw_help(self, p: QPainter, r: QRectF):
        k = min(max(self.help.value, 0.0), 1.0)
        p.setOpacity(k)
        p.fillRect(r, QColor(2, 5, 7, 190))
        w, h = 470, 60 + 26 * len(SHORTCUTS)
        panel = QRectF(0, 0, w, h)
        panel.moveCenter(r.center() + QPointF(0, 16 * (1 - k)))
        p.setPen(QPen(with_alpha(ACCENT, 160), 1))
        p.setBrush(QColor(4, 10, 14, 245))
        p.drawRect(panel)
        p.setPen(QPen(ACCENT, 2))
        brackets(p, panel, 14)
        p.setFont(mono(10, bold=True, spacing=3.0))
        p.setPen(ACCENT)
        p.drawText(QPointF(panel.left() + 26, panel.top() + 36), "KEYBOARD")
        for i, (key, desc) in enumerate(SHORTCUTS):
            yy = panel.top() + 70 + i * 26
            p.setFont(mono(8.5, bold=True, spacing=1.6))
            p.setPen(QColor(TEXT))
            p.drawText(QPointF(panel.left() + 26, yy), key)
            p.setFont(mono(8.5, spacing=1.0))
            p.setPen(QColor(DIM))
            p.drawText(QPointF(panel.left() + 170, yy), desc)
        p.setOpacity(1)
