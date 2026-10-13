"""Main window: ambient background, viewport + stats on the left, tabbed sidebar on the right."""

import time
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QImage, QKeySequence, QPainter, QPen, QRegion, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from n0face.app import motion
from n0face.app.background import Ambient
from n0face.app.services import PreviewRequest, PreviewService
from n0face.app.style import DIM, FAINT, LINE_STRONG, LIVE, OK, TEXT, accent_color, graffiti, mono, spray_text, stylesheet
from n0face.app.viewport import Viewport
from n0face.app.widgets import (
    ACCENT,
    ActionButton,
    AccessoryTile,
    AnimatedStack,
    Callout,
    ChoiceRow,
    DropZone,
    FileRow,
    SectionHeader,
    Segmented,
    Slider,
    StatusRow,
    Swatch,
    TabBar,
    TickBox,
    Toggle,
    set_accent,
)
from n0face.app.worker import FileWorker, Worker
from n0face.media import IMAGE_EXTS, VIDEO_EXTS, collect_inputs, ffmpeg_path, media_kind
from n0face.modes import MODE_NAMES, MODES
from n0face.pipeline import Engine, Stats
from n0face.theme import THEMES
from n0face import glitch, plugins, wrap

CAMERA_SLOTS = 4
FILE_FILTER = "Photos and videos ({})".format(" ".join(f"*{e}" for e in sorted(IMAGE_EXTS | VIDEO_EXTS)))
SIDEBAR_WIDTH = 400
EDGE = 6  # px band at the window border that resizes the frameless window
AMBIENT_FPS = 8
DETECTING_COPY = ["ACQUIRING LANDMARKS", "SCANNING TILES FOR SMALL FACES", "BUILDING MASK"]


class TitleBar(QFrame):
    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window_ = window
        self.setFixedHeight(56)
        self.setMouseTracking(True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 0, 8, 0)
        lay.setSpacing(0)
        lay.addStretch()
        for text, name, slot, tip in (
            ("—", "WinBtn", window.showMinimized, "Minimize"),
            ("□", "WinBtn", window.toggle_maximized, "Maximize  (F11)"),
            ("✕", "CloseBtn", window.close, "Close"),
        ):
            b = QPushButton(text)
            b.setObjectName(name)
            b.setFont(mono(11))
            b.setToolTip(tip)
            b.clicked.connect(slot)
            lay.addWidget(b)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = 40
        # Wordmark, sprayed on: "n0" in the accent color, "face" in white.
        font = graffiti(25)
        x = 24 + spray_text(p, QPointF(24, y), "n0", font, ACCENT, -3)
        x += spray_text(p, QPointF(x, y - 1), "face", font, QColor(TEXT), -3, glow=30) + 26
        p.setPen(QPen(QColor(LINE_STRONG), 1))
        p.drawLine(QPointF(x, 16), QPointF(x, 40))
        p.setFont(mono(9, spacing=2.6))
        p.setPen(QColor(DIM))
        p.drawText(int(x + 24), y - 8, "LOCAL FACE PRIVACY")
        tag = "privacy is a choice"
        tag_font = graffiti(14)
        p.setFont(tag_font)
        w = p.fontMetrics().horizontalAdvance(tag)
        spray_text(p, QPointF(self.width() - SIDEBAR_WIDTH - 64 - w, y - 4), tag, tag_font, ACCENT, -5)
        p.setPen(QPen(QColor(LINE_STRONG), 1))
        p.drawLine(QPointF(0, self.height() - 1), QPointF(self.width(), self.height() - 1))

    def mouseMoveEvent(self, e):
        top = e.position().y() <= EDGE and not self.window_.isMaximized()
        self.setCursor(Qt.CursorShape.SizeVerCursor if top else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if e.position().y() <= EDGE and not self.window_.isMaximized():
            self.window_.windowHandle().startSystemResize(Qt.Edge.TopEdge)
        else:
            self.window_.windowHandle().startSystemMove()  # native drag, incl. Aero snap

    def mouseDoubleClickEvent(self, _):
        self.window_.toggle_maximized()


class StatsBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(48)
        self.items: list[tuple[str, str]] = []
        self.text = ""

    def set_items(self, items: list[tuple[str, str]]) -> None:
        if (items, "") != (self.items, self.text):
            self.items, self.text = items, ""
            self.update()

    def set_text(self, text: str) -> None:
        if text != self.text:
            self.text, self.items = text, []
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(LINE_STRONG), 1))
        p.setBrush(QColor(4, 9, 12, 170))
        p.drawRect(r)
        p.setFont(mono(8.5, spacing=1.8))
        y = int(r.center().y()) + 5
        if self.text:
            p.setPen(QColor(TEXT))
            p.drawText(24, y, self.text)
        else:
            x = 24
            for i, (label, value) in enumerate(self.items):
                p.setPen(QColor(DIM))
                p.drawText(x, y, label)
                x += p.fontMetrics().horizontalAdvance(label) + 16
                p.setPen(QColor(TEXT))
                p.drawText(x, y, value)
                x += p.fontMetrics().horizontalAdvance(value) + 24
                if i < len(self.items) - 1:
                    p.setPen(QPen(QColor(LINE_STRONG), 1))
                    p.drawLine(QPointF(x - 12, r.top() + 12), QPointF(x - 12, r.bottom() - 12))
        p.setPen(QPen(QColor(FAINT), 1))
        for i in range(14):
            xx = r.right() - 20 - i * 3 - (i % 3)
            p.drawLine(QPointF(xx, r.top() + 16), QPointF(xx, r.bottom() - 16))


def page() -> tuple[QScrollArea, QVBoxLayout]:
    content = QWidget()
    content.setObjectName("Page")
    lay = QVBoxLayout(content)
    lay.setContentsMargins(24, 24, 24, 24)
    lay.setSpacing(16)
    scroll = QScrollArea()
    scroll.setWidget(content)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return scroll, lay


def lock_icon(_kind: str, px: int, color: QColor) -> QImage:
    """A padlock, for locked PRO tiles."""
    img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(color, max(px / 16, 1.5)))
    w, h = px * 0.46, px * 0.34
    body = QRectF((px - w) / 2, px * 0.48, w, h)
    p.drawRoundedRect(body, 2, 2)
    p.drawArc(QRectF(px / 2 - w * 0.3, px * 0.2, w * 0.6, w * 0.7), 0, 180 * 16)
    p.drawLine(QPointF(px / 2 - w * 0.3, px * 0.2 + w * 0.35), QPointF(px / 2 - w * 0.3, px * 0.48))
    p.drawLine(QPointF(px / 2 + w * 0.3, px * 0.2 + w * 0.35), QPointF(px / 2 + w * 0.3, px * 0.48))
    p.end()
    return img


def open_pro_page() -> None:
    QDesktopServices.openUrl(QUrl(plugins.PRO_URL))


def label(text: str, name: str = "Dim", size: float = 8.5) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName(name)
    lbl.setFont(mono(size, spacing=1.2))
    lbl.setWordWrap(True)
    return lbl


class MainWindow(QWidget):
    def __init__(self, engine: Engine):
        super().__init__()
        self.engine = engine
        self.settings = QSettings("n0face", "n0face")
        self.worker: Worker | None = None
        self.file_worker: FileWorker | None = None
        self._preview_request = 0
        self.files: list[Path] = []
        self.selected: Path | None = None
        self.out_dir: Path | None = None
        self.source = "files"
        self._outputs: list[Path] = []
        self._file_log: list[str] = []
        self._failures: list[str] = []
        self.ambient = Ambient()
        self._t0 = time.perf_counter()

        reduced = self.settings.value("reduce_motion", motion.system_prefers_reduced_motion(), type=bool)
        motion.set_reduced_motion(reduced)
        self.ambient.frozen = reduced
        self.auto_rotate = self.settings.value("auto_rotate", True, type=bool) and not reduced
        self._restore_engine()

        self.setWindowTitle("n0face")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setMinimumSize(1100, 720)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.title = TitleBar(self)
        outer.addWidget(self.title)
        body = QHBoxLayout()
        body.setContentsMargins(24, 16, 24, 16)
        body.setSpacing(24)
        left = QVBoxLayout()
        left.setSpacing(16)
        self.viewport = Viewport(lambda: (self.engine.mode_name, self.engine.effect_name, self.engine.theme, self.auto_rotate))
        self.viewport.orbit.auto_enabled = self.auto_rotate
        if not self.auto_rotate:
            self.viewport.orbit.auto.snap(0)
        left.addWidget(self.viewport, 1)
        self.stats = StatsBar()
        left.addWidget(self.stats)
        body.addLayout(left, 1)
        self.sidebar = self._sidebar()
        body.addWidget(self.sidebar)
        outer.addLayout(body, 1)

        self.previews = PreviewService()
        self.previews.ready.connect(self._on_preview)
        self.previews.failed.connect(self._on_preview_failed)
        self.previews.detecting.connect(self._on_preview_detecting)
        self.previews.start()

        self._apply_theme(self.engine.theme, initial=True)
        self._show_mode()
        self._on_stats(Stats())
        self._set_camera_live(False)
        self.st_stream.set("OFF", DIM)
        self.st_net.set("BLOCKED", OK)
        self.st_rec.set("NEVER", OK)
        self.set_files([])
        self._set_source("files")
        self.tabs.set_current(int(self.settings.value("tab", 0, type=int)))

        self._preview_debounce = QTimer(self, singleShot=True, interval=120, timeout=self._refresh_preview)
        self._ambient_timer = QTimer(self, interval=1000 // AMBIENT_FPS, timeout=self._ambient_tick)
        self._ambient_timer.start()
        self._shortcuts()
        self._restore_geometry()

    # ---- persistence ------------------------------------------------------------------

    def _restore_engine(self) -> None:
        # Preferences only. No file paths or anything about your media is ever stored.
        theme = THEMES.get(self.settings.value("theme", "cyan", type=str), self.engine.theme)
        self.engine.set_theme(theme)
        mode = self.settings.value("mode", self.engine.mode_name, type=str)
        if mode in MODES:
            self.engine.set_mode(mode, self.settings.value("effect", None, type=str))
        self.engine.hud_enabled = self.settings.value("hud", True, type=bool)
        wrap.config.enabled = self.settings.value("wrap", False, type=bool)
        glitch.config.enabled = self.settings.value("glitch", False, type=bool)
        for key in ("intensity", "frequency", "length"):
            setattr(glitch.config, key, self.settings.value(f"glitch_{key}", 1.0, type=float))
        wrap.config.backing = self.settings.value("backing", "blur", type=str)

    def _restore_geometry(self) -> None:
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1480, 920)

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("theme", next(k for k, v in THEMES.items() if v == self.engine.theme))
        s.setValue("mode", self.engine.mode_name)
        s.setValue("effect", self.engine.effect_name)
        s.setValue("hud", self.engine.hud_enabled)
        s.setValue("wrap", wrap.config.enabled)
        s.setValue("glitch", glitch.config.enabled)
        for key in ("intensity", "frequency", "length"):
            s.setValue(f"glitch_{key}", getattr(glitch.config, key))
        s.setValue("backing", wrap.config.backing)
        s.setValue("auto_rotate", self.rotate_toggle.isChecked())
        s.setValue("reduce_motion", motion.reduced_motion())
        s.setValue("tab", self.tabs.current)
        s.setValue("geometry", self.saveGeometry())

    # ---- background -------------------------------------------------------------------

    def _ambient_tick(self) -> None:
        if self.isMinimized() or not self.isVisible():
            return
        self.ambient.refresh(self.size(), self.devicePixelRatioF(), ACCENT, time.perf_counter() - self._t0)
        if self.ambient.frozen:
            return
        # Repaint everything except the sidebar: it's a near-opaque panel, so redrawing its
        # widgets just to move a glow behind them would be wasted work.
        region = QRegion(self.rect()) - QRegion(self.sidebar.geometry())
        self.update(region)

    def paintEvent(self, e):
        if self.ambient.frame is None:
            self.ambient.refresh(self.size(), self.devicePixelRatioF(), ACCENT, 0.0)
        p = QPainter(self)
        self.ambient.paint(p, QRectF(e.rect()))

    def resizeEvent(self, e):
        self.ambient.refresh(self.size(), self.devicePixelRatioF(), ACCENT, time.perf_counter() - self._t0)
        size = self.viewport.image_area()
        for w in (self.worker, self.file_worker):
            if w is not None:
                w.display_size = (size.width(), size.height())
        super().resizeEvent(e)

    # ---- frameless window behavior ----------------------------------------------------

    def toggle_maximized(self) -> None:
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def _edges_at(self, pos) -> Qt.Edge:
        edges = Qt.Edge(0)
        if self.isMaximized():
            return edges
        if pos.x() <= EDGE:
            edges |= Qt.Edge.LeftEdge
        if pos.x() >= self.width() - EDGE:
            edges |= Qt.Edge.RightEdge
        if pos.y() >= self.height() - EDGE:
            edges |= Qt.Edge.BottomEdge
        return edges

    def mouseMoveEvent(self, e):
        edges = self._edges_at(e.position())
        horizontal = bool(edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge))
        vertical = bool(edges & Qt.Edge.BottomEdge)
        if horizontal and vertical:
            shape = Qt.CursorShape.SizeFDiagCursor if edges & Qt.Edge.RightEdge else Qt.CursorShape.SizeBDiagCursor
        elif horizontal:
            shape = Qt.CursorShape.SizeHorCursor
        elif vertical:
            shape = Qt.CursorShape.SizeVerCursor
        else:
            shape = Qt.CursorShape.ArrowCursor
        self.setCursor(shape)

    def mousePressEvent(self, e):
        edges = self._edges_at(e.position())
        if e.button() == Qt.MouseButton.LeftButton and edges:
            self.windowHandle().startSystemResize(edges)

    # ---- keyboard ---------------------------------------------------------------------

    def _shortcuts(self) -> None:
        def bind(keys: str, fn) -> None:
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(fn)

        bind("Ctrl+O", self._pick_files)
        bind("Ctrl+Return", self._primary)
        bind("Ctrl+Enter", self._primary)
        bind("Escape", self._escape)
        bind("F11", self.toggle_maximized)
        bind("M", lambda: self._cycle_mode())
        bind("R", self.viewport.reset_view)
        bind("?", lambda: self.viewport.toggle_help())
        bind("Shift+/", lambda: self.viewport.toggle_help())
        for i in range(len(self.tabs.labels)):
            bind(f"Ctrl+{i + 1}", lambda i=i: self.tabs.set_current(i))
        for i in range(9):
            bind(str(i + 1), lambda i=i: self._effect_by_index(i))

    def _primary(self) -> None:
        if self.engage.isEnabled():
            self.engage.click()

    def _escape(self) -> None:
        if self.viewport.help.target > 0.5:
            self.viewport.toggle_help(False)
        elif self.file_worker is not None or self.worker is not None:
            self._toggle_running()

    def _cycle_mode(self) -> None:
        if self.file_worker is None:
            i = MODE_NAMES.index(self.engine.mode_name)
            self._select_mode(MODE_NAMES[(i + 1) % len(MODE_NAMES)])

    def _effect_by_index(self, i: int) -> None:
        effects = MODES[self.engine.mode_name].effects
        if i < len(effects) and self.file_worker is None:
            self._select_effect(effects[i])

    # ---- sidebar ----------------------------------------------------------------------

    def _sidebar(self) -> QWidget:
        side = QFrame()
        side.setObjectName("Sidebar")
        side.setFixedWidth(SIDEBAR_WIDTH)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.tabs = TabBar(["SOURCE", "MASK", "GEAR", "LOOK", "SYSTEM"])
        lay.addWidget(self.tabs)
        self.pages = AnimatedStack()
        for build in (self._page_source, self._page_mask, self._page_gear, self._page_look, self._page_system):
            self.pages.addWidget(build())
        self.tabs.changed.connect(self.pages.slide_to)
        lay.addWidget(self.pages, 1)

        bottom = QVBoxLayout()
        bottom.setContentsMargins(24, 16, 24, 24)
        bottom.setSpacing(8)
        self.summary = label("", "Faint", 7.5)
        bottom.addWidget(self.summary)
        self.engage = ActionButton()
        self.engage.setToolTip("Ctrl+Enter")
        self.engage.clicked.connect(self._toggle_running)
        bottom.addWidget(self.engage)
        lay.addLayout(bottom)
        return side

    def _page_source(self) -> QWidget:
        scroll, lay = page()
        lay.addWidget(SectionHeader(1, "SOURCE"))
        self.source_seg = Segmented([("files", "FILES"), ("camera", "CAMERA")])
        self.source_seg.changed.connect(self._set_source)
        lay.addWidget(self.source_seg)

        self.source_pages = AnimatedStack()
        files = QWidget()
        fl = QVBoxLayout(files)
        fl.setContentsMargins(0, 8, 0, 0)
        fl.setSpacing(8)
        self.drop = DropZone()
        self.drop.setToolTip("Ctrl+O")
        self.drop.clicked.connect(self._pick_files)
        fl.addWidget(self.drop)
        self.file_group = QButtonGroup(self)
        self.file_list = QVBoxLayout()
        self.file_list.setSpacing(8)
        fl.addLayout(self.file_list)
        fl.addSpacing(16)

        fl.addWidget(SectionHeader(2, "OUTPUT FILES"))
        fl.addSpacing(8)
        save = QHBoxLayout()
        save.setSpacing(8)
        save_lbl = label("SAVE TO", "Dim", 8.5)
        save_lbl.setFixedWidth(72)
        save.addWidget(save_lbl)
        self.out_field = QLineEdit()
        self.out_field.setReadOnly(True)
        self.out_field.setFont(mono(8.5, spacing=0.8))
        save.addWidget(self.out_field, 1)
        folder = QPushButton("❐")
        folder.setObjectName("IconBtn")
        folder.setFont(mono(11))
        folder.setFixedWidth(40)
        folder.setToolTip("Choose output folder (originals are never changed)")
        folder.clicked.connect(self._pick_out_dir)
        save.addWidget(folder)
        fl.addLayout(save)
        fl.addSpacing(8)
        self.hide_miss = TickBox("HIDE IF NO FACE FOUND")
        self.hide_miss.setChecked(True)
        self.hide_miss.setToolTip("Fail-closed: a photo or frame with no detected face is hidden, not passed through.")
        self.hide_miss.toggled.connect(lambda _: self._schedule_preview())
        fl.addWidget(self.hide_miss)
        self.keep_audio = TickBox("KEEP AUDIO", "(voice not hidden)")
        if ffmpeg_path() is None:
            self.keep_audio.setEnabled(False)
            self.keep_audio.setToolTip("Needs ffmpeg on PATH")
        fl.addWidget(self.keep_audio)
        fl.addStretch()
        self.source_pages.addWidget(files)

        cam = QWidget()
        cl = QVBoxLayout(cam)
        cl.setContentsMargins(0, 8, 0, 0)
        cl.setSpacing(16)
        self.camera_box = QComboBox()
        self.camera_box.setFont(mono(9, spacing=1.4))
        self.camera_box.addItems([f"CAMERA {i}" for i in range(CAMERA_SLOTS)])
        cl.addWidget(self.camera_box)
        self.stream_toggle = Toggle("VIRTUAL CAMERA")
        self.stream_toggle.setChecked(True)
        self.stream_toggle.toggled.connect(self._toggle_stream)
        cl.addWidget(self.stream_toggle)
        cl.addWidget(Callout(
            "Shows up as “OBS Virtual Camera” in Discord, Teams, Zoom and OBS. "
            "Needs OBS Studio installed (it doesn't have to be running).", "info"))
        cl.addStretch()
        self.source_pages.addWidget(cam)
        lay.addWidget(self.source_pages, 1)
        return scroll

    def _page_mask(self) -> QWidget:
        scroll, lay = page()
        lay.addWidget(SectionHeader(3, "MODE"))
        self.mode_seg = Segmented([(m, m.upper()) for m in MODES])
        self.mode_seg.setToolTip("M cycles modes")
        self.mode_seg.changed.connect(self._select_mode)
        lay.addWidget(self.mode_seg)
        self.claim = Callout()
        lay.addWidget(self.claim)
        lay.addSpacing(16)
        lay.addWidget(SectionHeader(4, "EFFECT"))
        self.effect_group = QButtonGroup(self)
        self.locked_rows: list[ChoiceRow] = []
        self.effect_box = QVBoxLayout()
        self.effect_box.setSpacing(8)
        lay.addLayout(self.effect_box)
        lay.addSpacing(8)
        self.wrap_toggle = Toggle("3D WRAP", "(hugs the face)")
        self.wrap_toggle.setToolTip("The mask follows your face in 3D; hair and ears stay visible.")
        self.wrap_toggle.setChecked(wrap.config.enabled)
        self.wrap_toggle.toggled.connect(self._toggle_wrap)
        lay.addWidget(self.wrap_toggle)
        self.glitch_toggle = Toggle("GLITCH", "(random bursts)")
        self.glitch_toggle.setToolTip("Every couple of seconds the mask tears, warps and splits for a moment.")
        self.glitch_toggle.setChecked(glitch.config.enabled)
        self.glitch_toggle.toggled.connect(self._toggle_glitch)
        lay.addWidget(self.glitch_toggle)
        # Its knobs (n0face PRO) show while it's on.
        self.glitch_opts = QWidget()
        opts = QVBoxLayout(self.glitch_opts)
        opts.setContentsMargins(52, 0, 0, 4)
        opts.setSpacing(2)
        knobs = () if not plugins.has("glitch_knobs") else (
            ("intensity", "INTENSITY", 0.25, 2.0, "How hard each burst tears the mask."),
            ("frequency", "FREQUENCY", 0.25, 3.0, "How often bursts come (1x: about one every two seconds)."),
            ("length", "LENGTH", 0.5, 2.0, "How long each burst lasts."),
        )
        for key, name, lo, hi, tip in knobs:
            row = QHBoxLayout()
            name_lbl = label(name, "Dim", 7.5)
            name_lbl.setFixedWidth(84)
            row.addWidget(name_lbl)
            slider = Slider(lo, hi, getattr(glitch.config, key), lambda v: f"{v:.2f}x")
            slider.setToolTip(tip)
            slider.changed.connect(lambda v, k=key: self._set_glitch_option(k, v))
            row.addWidget(slider, 1)
            opts.addLayout(row)
        if not knobs:
            locked = QPushButton("INTENSITY \u00b7 FREQUENCY \u00b7 LENGTH   PRO")
            locked.setObjectName("Locked")
            locked.setFont(mono(7.5, spacing=1.4))
            locked.setToolTip("Tune glitch bursts with n0face PRO.")
            locked.clicked.connect(open_pro_page)
            opts.addWidget(locked)
        self.glitch_opts.setVisible(glitch.config.enabled)
        lay.addWidget(self.glitch_opts)
        self.hud_toggle = Toggle("STATUS HUD", "(ID / CONF box)")
        self.hud_toggle.setToolTip("The ID / CONF / STATUS MASKED box, bounding box and tracking lines.")
        self.hud_toggle.setChecked(self.engine.hud_enabled)
        self.hud_toggle.toggled.connect(self._toggle_hud)
        lay.addWidget(self.hud_toggle)
        lay.addSpacing(8)
        lay.addWidget(label("UNDER THE MASK", "Dim", 7.5))
        self.backing_seg = Segmented([("blur", "BLUR"), ("solid", "SOLID")])
        self.backing_seg.set_value(wrap.config.backing, animate=False)
        self.backing_seg.setToolTip("Mask mode: a heavy blur of your face under the design, or a solid backing.")
        self.backing_seg.changed.connect(self._set_backing)
        lay.addWidget(self.backing_seg)
        lay.addStretch()
        return scroll

    def _page_gear(self) -> QWidget:
        build = plugins.registry.pages.get("gear")
        if build is not None:
            return build(self)
        scroll, lay = page()
        lay.addWidget(SectionHeader(5, "ACCESSORIES"))
        lay.addWidget(label("n0face PRO. Horns, halo, crown and more, riding on your head in 3D.", "Dim", 7.5))
        grid = QGridLayout()
        grid.setSpacing(8)
        for i, kind in enumerate(plugins.PRO_ACCESSORIES):
            tile = AccessoryTile(kind, lock_icon)
            tile.setCheckable(False)
            tile.setToolTip(f"{kind.upper()} \u00b7 n0face PRO")
            tile.clicked.connect(open_pro_page)
            grid.addWidget(tile, i // 3, i % 3)
        lay.addLayout(grid)
        lay.addSpacing(16)
        lay.addWidget(SectionHeader(6, "CUSTOMIZE"))
        lay.addWidget(label(
            "n0face PRO. Recolour each part of a mask (eyes, mouth, lines, accents, base), swap "
            "eye and mouth styles, set stroke and glow, tune GLITCH, and get five more masks: "
            "LED, CIRCUIT, HEX, SKULL, WIREFRAME.", "Dim", 7.5))
        get = QPushButton("GET n0face PRO")
        get.setObjectName("Locked")
        get.setFont(mono(9, spacing=2.4))
        get.clicked.connect(open_pro_page)
        lay.addWidget(get)
        lay.addStretch()
        return scroll

    def _page_look(self) -> QWidget:
        scroll, lay = page()
        lay.addWidget(SectionHeader(7, "THEME"))
        row = QHBoxLayout()
        row.setSpacing(8)
        self.theme_group = QButtonGroup(self)
        for name, theme in THEMES.items():
            sw = Swatch(accent_color(theme), name)
            sw.setChecked(theme == self.engine.theme)
            sw.clicked.connect(lambda _=False, t=theme: self._apply_theme(t))
            self.theme_group.addButton(sw)
            row.addWidget(sw)
        row.addStretch()
        lay.addLayout(row)
        lay.addSpacing(16)
        lay.addWidget(SectionHeader(8, "PREVIEW"))
        self.rotate_toggle = Toggle("AUTO-ROTATE PREVIEW")
        self.rotate_toggle.setChecked(self.auto_rotate)
        self.rotate_toggle.toggled.connect(self._toggle_rotate)
        lay.addWidget(self.rotate_toggle)
        lay.addSpacing(16)
        lay.addWidget(SectionHeader(9, "MOTION"))
        self.motion_toggle = Toggle("REDUCE MOTION")
        self.motion_toggle.setChecked(motion.reduced_motion())
        self.motion_toggle.setToolTip("Turns off transitions, drifting glows and idle animation.")
        self.motion_toggle.toggled.connect(self._toggle_reduced_motion)
        lay.addWidget(self.motion_toggle)
        lay.addWidget(label(
            "The 3D preview uses a generic, anonymous face model. It is nobody's face. "
            "Press ? for keyboard shortcuts."))
        lay.addStretch()
        return scroll

    def _page_system(self) -> QWidget:
        scroll, lay = page()
        lay.addWidget(SectionHeader(10, "STATUS"))
        self.st_cam = StatusRow("CAMERA")
        self.st_stream = StatusRow("VIRTUAL CAM")
        self.st_net = StatusRow("NETWORK")
        self.st_rec = StatusRow("RECORDING")
        for w in (self.st_cam, self.st_stream, self.st_net, self.st_rec):
            lay.addWidget(w)
        self.stream_note = label("", "Dim", 7.5)
        lay.addWidget(self.stream_note)
        lay.addSpacing(16)
        lay.addWidget(SectionHeader(11, "LAST RUN"))
        self.run_log = label("nothing processed yet", "Dim", 7.5)
        lay.addWidget(self.run_log)
        self.open_folder = QPushButton("OPEN OUTPUT FOLDER")
        self.open_folder.setFont(mono(8.5, spacing=1.8))
        self.open_folder.hide()
        self.open_folder.clicked.connect(self._open_output_folder)
        lay.addWidget(self.open_folder)
        lay.addSpacing(16)
        lay.addWidget(SectionHeader(12, "ABOUT"))
        lay.addWidget(label(
            "Face and head detection run on this device with two local models. Network access is "
            "disabled while n0face runs. Nothing is uploaded, logged or recorded. Only display "
            "preferences are remembered between sessions, never file names or paths."))
        lay.addStretch()
        return scroll

    # ---- mode / effect / look ---------------------------------------------------------

    def _select_mode(self, name: str) -> None:
        if name == self.engine.mode_name:
            return
        self.engine.set_mode(name)
        self._show_mode()
        self.viewport.style_changed()
        self._schedule_preview()

    def _show_mode(self) -> None:
        mode = MODES[self.engine.mode_name]
        self.mode_seg.set_value(mode.name)
        self.claim.set(mode.claim, "warn" if mode.name == "mask" else "info")
        for b in self.effect_group.buttons():
            self.effect_group.removeButton(b)
            self.effect_box.removeWidget(b)
            b.hide()
            b.deleteLater()
        for i, name in enumerate(mode.effects, 1):
            row = ChoiceRow(i, name.upper(), "weakest" if name == "pixelate" else "")
            row.setToolTip(f"key {i}")
            row.setChecked(name == self.engine.effect_name)
            row.clicked.connect(lambda _=False, n=name: self._select_effect(n))
            self.effect_group.addButton(row)
            self.effect_box.addWidget(row)
        for b in self.locked_rows:
            self.effect_box.removeWidget(b)
            b.deleteLater()
        self.locked_rows = []
        if mode.name == "mask":
            for name in (n for n in plugins.PRO_MASKS if n not in mode.effects):
                row = ChoiceRow("·", name.upper(), "PRO")
                row.setCheckable(False)
                row.setEnabled(True)
                row.setToolTip("n0face PRO")
                row.clicked.connect(open_pro_page)
                self.locked_rows.append(row)
                self.effect_box.addWidget(row)

    def _select_effect(self, name: str) -> None:
        if name == self.engine.effect_name:
            return
        self.engine.set_effect(name)
        for b in self.effect_group.buttons():
            if isinstance(b, ChoiceRow) and b.label == name.upper() and not b.isChecked():
                b.setChecked(True)
        self.viewport.style_changed()
        self._schedule_preview()

    def _apply_theme(self, theme, initial: bool = False) -> None:
        self.engine.set_theme(theme)
        set_accent(accent_color(theme))
        for hook in getattr(self, "theme_hooks", []):  # add-on pages that follow the theme
            hook(theme)
        if initial or not hasattr(self, "_styled_for") or self._styled_for != theme:
            self.setStyleSheet(stylesheet(theme))
            self._styled_for = theme
        self.ambient.refresh(self.size(), self.devicePixelRatioF(), ACCENT, time.perf_counter() - self._t0)
        if not initial:
            self.viewport.style_changed()
            self._schedule_preview()
        self.update()

    def _toggle_wrap(self, on: bool) -> None:
        wrap.config.enabled = on
        self.viewport.style_changed()
        self._schedule_preview()

    def _set_backing(self, backing: str) -> None:
        wrap.config.backing = backing
        self.viewport.style_changed()
        self._schedule_preview()

    def _toggle_glitch(self, on: bool) -> None:
        glitch.config.enabled = on
        self.glitch_opts.setVisible(on)
        self._schedule_preview()

    def _set_glitch_option(self, key: str, value: float) -> None:
        setattr(glitch.config, key, value)
        self._schedule_preview()

    def _toggle_hud(self, on: bool) -> None:
        self.engine.hud_enabled = on
        self._schedule_preview()

    def _toggle_rotate(self, on: bool) -> None:
        self.auto_rotate = on
        self.viewport.set_auto_rotate(on)

    def _toggle_reduced_motion(self, on: bool) -> None:
        motion.set_reduced_motion(on)
        self.ambient.frozen = on
        self.ambient.refresh(self.size(), self.devicePixelRatioF(), ACCENT, time.perf_counter() - self._t0)
        if on and self.rotate_toggle.isChecked():
            self.rotate_toggle.setChecked(False)
        self.update()

    # ---- source -----------------------------------------------------------------------

    def _set_source(self, name: str) -> None:
        self.source = name
        self.source_seg.set_value(name)
        self.source_pages.slide_to(0 if name == "files" else 1)
        if name == "files":
            self.viewport.caption = "OUTPUT // PROCESSED COPY PREVIEW"
            self.viewport.message = ("NO FILES", "drop photos / videos here, or press ctrl+O")
            self._schedule_preview()
        else:
            self.viewport.caption = "OUTPUT // WHAT OTHER APPS RECEIVE"
            self.viewport.message = ("CAMERA OFF", "press ENGAGE — the feed is masked from frame one")
            self.viewport.busy_text = ""
            self.viewport.set_image(None)
        self.viewport.style_changed()
        self._refresh_engage()

    def _refresh_engage(self) -> None:
        if self.source == "files":
            n = len(self.files)
            self.engage.setText(f"PROCESS {n} FILE{'S' if n != 1 else ''}" if n else "SELECT FILES")
            self.engage.setEnabled(n > 0)
            self.summary.setText("copies are saved with all metadata removed; originals are never changed")
        else:
            self.engage.setText("ENGAGE")
            self.engage.setEnabled(True)
            self.summary.setText("the camera is only on while engaged")

    # ---- files ------------------------------------------------------------------------

    def _pick_files(self) -> None:
        if self.file_worker is not None or self.worker is not None:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Select photos and videos", "", FILE_FILTER)
        if paths:
            self._set_source("files")
            self.tabs.set_current(0)
            self.add_files([Path(p) for p in paths])

    def add_files(self, paths: list[Path]) -> None:
        before = set(self.files)
        self.set_files(self.files + paths)
        added = [f for f in self.files if f not in before]
        skipped = len(paths) - len(added)
        if added and self.selected not in added:
            self._select_file(added[0])
        if skipped > 0 and not added:
            self.viewport.toast("NOTHING NEW TO ADD", "those files are already listed, or aren't photos/videos", "warn")

    def set_files(self, paths: list[Path]) -> None:
        seen: set[Path] = set()
        old = set(self.files)
        self.files = [f for f in collect_inputs(paths) if not (f in seen or seen.add(f))]
        self.previews.forget(set(self.files))
        while self.file_list.count():
            item = self.file_list.takeAt(0)
            if item.widget():
                self.file_group.removeButton(item.widget())
                item.widget().hide()
                item.widget().deleteLater()
        stagger = 0
        for f in self.files:
            delay = 0 if f in old else 45 * stagger
            stagger += f not in old
            row = FileRow(f, media_kind(f), delay)
            if f in old:
                row.appear.snap(1)
            row.removed.connect(self._remove_file)
            row.selected.connect(self._select_file)
            self.file_group.addButton(row)
            self.file_list.addWidget(row)
        if self.selected not in self.files:
            self.selected = self.files[0] if self.files else None
        self._mark_selected()
        self._update_out_field()
        self._refresh_engage()
        self._schedule_preview()

    def _remove_file(self, path: Path) -> None:
        self.set_files([f for f in self.files if f != path])

    def _select_file(self, path: Path) -> None:
        if path != self.selected:
            self.selected = path
            self._mark_selected()
            self._schedule_preview()

    def _mark_selected(self) -> None:
        for b in self.file_group.buttons():
            if isinstance(b, FileRow) and b.path == self.selected:
                b.setChecked(True)

    def _pick_out_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Save processed copies to")
        self.out_dir = Path(folder) if folder else None
        self._update_out_field()

    def _update_out_field(self) -> None:
        self.out_field.setText(str(self.out_dir) if self.out_dir else "next to originals")

    def _schedule_preview(self) -> None:
        if hasattr(self, "_preview_debounce"):
            self._preview_debounce.start()

    def _refresh_preview(self) -> None:
        if self.source != "files" or self.file_worker is not None:
            return
        if self.selected is None:
            self.viewport.busy_text = ""
            self.viewport.resolution = "--"
            self.viewport.set_image(None)
            return
        self._preview_request += 1
        self.previews.submit(PreviewRequest(
            self._preview_request, self.selected, self.engine.mode_name, self.engine.effect_name,
            self.engine.theme, self.engine.hud_enabled, "hide" if self.hide_miss.isChecked() else "keep",
        ))

    def _on_preview_detecting(self, request: int) -> None:
        if request == self._preview_request:
            self.viewport.busy_text = DETECTING_COPY

    def _on_preview(self, request: int, img, path: str) -> None:
        if request == self._preview_request and self.source == "files" and self.file_worker is None:
            # A new file reveals with a scan line; restyling the same file just swaps.
            new_file = getattr(self, "_previewed", None) != path or self.viewport.image is None
            self._previewed = path
            self.viewport.busy_text = ""
            self.viewport.set_image(img, reveal=new_file)

    def _on_preview_failed(self, request: int, msg: str) -> None:
        if request == self._preview_request:
            self.viewport.busy_text = ""
            name = self.selected.name if self.selected else "this file"
            self.viewport.toast(
                f"CAN'T PREVIEW {name.upper()}", "it may be damaged, or in a format OpenCV can't read", "error", 6
            )

    def _start_files(self) -> None:
        self.file_worker = FileWorker(
            self.files, self.out_dir, self.engine.mode_name, self.engine.effect_name, self.engine.theme,
            self.engine.hud_enabled, "hide" if self.hide_miss.isChecked() else "keep", self.keep_audio.isChecked(),
        )
        size = self.viewport.image_area()
        self.file_worker.display_size = (size.width(), size.height())
        self.file_worker.frame_ready.connect(self.viewport.set_image)
        self.file_worker.progress.connect(self._on_file_progress)
        self.file_worker.file_done.connect(self._on_file_done)
        self.file_worker.file_failed.connect(self._on_file_failed)
        self.file_worker.all_done.connect(self._on_files_done)
        self._file_log, self._outputs, self._failures = [], [], []
        self.open_folder.hide()
        self._lock_controls(True)
        self.engage.setText("CANCEL")
        self.engage.set_running(True)
        self.file_worker.start()

    def _lock_controls(self, locked: bool) -> None:
        for w in (self.source_seg, self.mode_seg, self.drop, self.hud_toggle, self.hide_miss, self.keep_audio):
            w.setEnabled(not locked)
        for group in (self.effect_group, self.theme_group, self.file_group):
            for b in group.buttons():
                b.setEnabled(not locked)

    def _on_file_progress(self, i: int, n: int, frac: float, name: str) -> None:
        self.engage.set_progress((i - 1 + frac) / max(n, 1))
        bar = "█" * int(frac * 24)
        self.stats.set_text(f"FILE {i}/{n}    {name}    {bar:░<24}  {frac * 100:5.1f}%")

    def _on_file_done(self, report) -> None:
        self._outputs.append(report.output)
        line = report.summary()
        if report.kept_unmasked_frames:
            line += "  !! content kept UNMASKED"
        self._file_log.append(line)

    def _on_file_failed(self, name: str, error: str) -> None:
        self._file_log.append(f"{name}: FAILED: {error}")
        self._failures.append(name)

    def _on_files_done(self, ok: int, failed: int, cancelled: bool) -> None:
        self.file_worker.wait()
        self.file_worker = None
        self._lock_controls(False)
        self.engage.set_running(False)
        self._refresh_engage()
        state = "CANCELLED" if cancelled else "DONE"
        self.stats.set_text(f"{state}    saved {ok}    failed {failed}    → details in SYSTEM tab")
        self.run_log.setText("\n".join(self._file_log[-6:]) or "nothing saved")
        self.open_folder.setVisible(bool(self._outputs))
        if ok and not cancelled:
            self.engage.celebrate()
            self.viewport.toast(
                f"{ok} FILE{'S' if ok != 1 else ''} SECURED",
                "masked copies saved · metadata stripped · originals untouched", "ok", 5,
            )
        if failed:
            self.viewport.toast(
                f"{failed} FILE{'S' if failed != 1 else ''} COULDN'T BE PROCESSED",
                f"{', '.join(self._failures[:2])} — see SYSTEM tab for the reason", "error", 7,
            )
        if cancelled:
            self.viewport.toast("CANCELLED", "partial outputs were deleted", "warn", 4)
        self._schedule_preview()

    def _open_output_folder(self) -> None:
        if self._outputs:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._outputs[-1].parent)))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and self.file_worker is None and self.worker is None:
            self.drop.set_drag_active(True)
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self.drop.set_drag_active(False)

    def dropEvent(self, e):
        self.drop.set_drag_active(False)
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        self._set_source("files")
        self.tabs.set_current(0)
        self.add_files(paths)

    # ---- running ----------------------------------------------------------------------

    def _toggle_running(self) -> None:
        if self.file_worker is not None:
            self.file_worker.stop()
            self.engage.setText("CANCELLING")
        elif self.worker is not None:
            self.worker.stop()
        elif self.source == "files":
            if self.files:
                self._start_files()
        else:
            self._start_camera()

    def _toggle_stream(self, on: bool) -> None:
        if self.worker is not None:
            self.worker.want_stream = on

    def _start_camera(self) -> None:
        self.worker = Worker(self.engine, self.camera_box.currentIndex(), self.stream_toggle.isChecked())
        size = self.viewport.image_area()
        self.worker.display_size = (size.width(), size.height())
        self.worker.frame_ready.connect(self.viewport.set_image)
        self.worker.stats_ready.connect(self._on_stats)
        self.worker.camera_state.connect(self._set_camera_live)
        self.worker.stream_state.connect(self._on_stream_state)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_camera_finished)
        self.worker.start()
        self.camera_box.setEnabled(False)
        self.source_seg.setEnabled(False)
        self.engage.setText("DISENGAGE")
        self.engage.set_running(True)
        self.viewport.busy_text = ["WAKING CAMERA", "LOADING FACE MODEL", "FEED HIDDEN UNTIL A FACE LOCKS"]

    def _on_camera_finished(self) -> None:
        self.worker = None
        self.viewport.hint = ""
        self.viewport.busy_text = ""
        self.viewport.set_image(None)
        self.viewport.fps = "--"
        self.camera_box.setEnabled(True)
        self.source_seg.setEnabled(True)
        self.engage.set_running(False)
        self._refresh_engage()
        self._on_stats(Stats())

    def _set_camera_live(self, live: bool) -> None:
        self.viewport.cam_live = live
        self.st_cam.set("LIVE" if live else "OFF", LIVE if live else DIM, live=live)

    def _on_stream_state(self, on: bool, info: str) -> None:
        self.st_stream.set("ON" if on else "OFF", OK if on else DIM)
        self.stream_note.setText(f"publishing as: {info}" if on else info)
        if on:
            self.viewport.toast("VIRTUAL CAMERA ON", f"pick “{info}” in Discord / Teams / Zoom / OBS", "ok")
        elif info:
            self.stream_toggle.blockSignals(True)
            self.stream_toggle.setChecked(False)
            self.stream_toggle.blockSignals(False)
            self.viewport.toast(
                "VIRTUAL CAMERA UNAVAILABLE",
                "the preview here still works \u00b7 install OBS Studio to use n0face in Discord / Teams / Zoom",
                "info", 8,
            )

    def _on_stats(self, s: Stats) -> None:
        if self.worker is None:
            self.stats.set_items([(k, "--") for k in ("FPS", "DETECT", "RENDER", "FACES", "LOCK", "FAIL-CLOSED FRAMES")])
            return
        self.viewport.fps = f"{s.fps:.1f}" if s.fps else "--"
        # The camera is on but nothing is shown until a face locks (fail-closed). Say so,
        # so a hidden feed never reads as a broken camera.
        self.viewport.hint = (
            ["CAMERA ON \u00b7 LOOKING FOR YOUR FACE", "STEP INTO FRAME, FACE THE CAMERA, ADD LIGHT",
             "THE FEED STAYS HIDDEN UNTIL YOUR FACE IS FOUND"]
            if s.lost else ""
        )
        self.stats.set_items([
            ("FPS", f"{s.fps:.1f}"),
            ("DETECT", f"{s.detect_ms:.1f}ms"),
            ("RENDER", f"{s.render_ms:.1f}ms"),
            ("FACES", str(s.faces)),
            ("LOCK", "LOST" if s.lost else "OK"),
            ("FAIL-CLOSED FRAMES", str(s.fail_closed_frames)),
        ])

    def _on_failed(self, msg: str) -> None:
        self.viewport.toast("CAMERA PROBLEM", msg[:90], "error", 7)

    def closeEvent(self, e):
        self._save_settings()
        self._ambient_timer.stop()
        for worker in (self.worker, self.file_worker):
            if worker is not None and worker.isRunning():
                worker.stop()
                worker.wait(5000)
        self.previews.stop()
        self.previews.wait(3000)
        self.viewport.shutdown()
        super().closeEvent(e)
