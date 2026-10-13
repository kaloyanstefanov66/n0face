"""Background threads that keep heavy work off the UI thread.

- RenderThread: renders 3D preview frames. The UI submits the newest request; if a frame is
  already being rendered, older pending requests are simply replaced (latest wins), so the
  queue can never build up lag.
- PreviewService: one long-lived face detector for file previews, plus a small in-memory
  cache of each file's frame and detections. Changing mode/effect/theme re-renders from the
  cache without re-running detection. Nothing is written to disk.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from n0face.app.preview3d import MaskScene, Pose
from n0face.theme import Theme

PREVIEW_MAX_SIDE = 1600
CACHE_FILES = 8


def bgr_to_qimage(img: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def fit_size(w: int, h: int, max_w: int, max_h: int) -> tuple[int, int]:
    s = min(max_w / w, max_h / h, 1.0)
    return max(int(w * s), 1), max(int(h * s), 1)


@dataclass(frozen=True)
class RenderRequest:
    width: int
    height: int
    effect: str
    mode: str
    theme: Theme
    t: float
    pose: Pose
    hover: tuple[float, float] | None
    px: float


class RenderThread(QThread):
    """Renders the newest RenderRequest; emits (QImage, frame metadata)."""

    rendered = Signal(object, object)  # QImage, Frame (image field dropped)

    def __init__(self, scene: MaskScene):
        super().__init__()
        self.scene = scene
        self._cond = threading.Condition()
        self._pending: RenderRequest | None = None
        self._stop = False
        self.busy = False

    def submit(self, request: RenderRequest) -> None:
        with self._cond:
            self._pending = request
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()

    def run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                req, self._pending = self._pending, None
                self.busy = True
            frame = self.scene.render(
                req.width, req.height, req.effect, req.mode, req.theme, req.t, req.pose, req.hover, req.px
            )
            h, w = frame.image.shape[:2]
            img = QImage(frame.image.data, w, h, 4 * w, QImage.Format.Format_ARGB32_Premultiplied).copy()
            frame.image = None
            self.busy = False
            self.rendered.emit(img, frame)


@dataclass(frozen=True)
class PreviewRequest:
    request: int
    path: Path
    mode: str
    effect: str
    theme: Theme
    hud: bool
    on_miss: str


class PreviewService(QThread):
    ready = Signal(int, QImage, str)  # request id, processed preview, file path
    failed = Signal(int, str)
    detecting = Signal(int)  # request id: detection (the slow part) has started

    def __init__(self):
        super().__init__()
        self._cond = threading.Condition()
        self._pending: PreviewRequest | None = None
        self._stop = False
        self._cache: OrderedDict[Path, tuple[np.ndarray, list[np.ndarray]]] = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, request: PreviewRequest) -> None:
        with self._cond:
            self._pending = request
            self._cond.notify()

    def forget(self, keep: set[Path]) -> None:
        """Drop cached frames for files no longer in the list."""
        with self._lock:
            for path in list(self._cache):
                if path not in keep:
                    del self._cache[path]

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()

    def run(self) -> None:
        from n0face.detector import StillDetector
        from n0face.media import MediaProcessor, first_frame

        detector = None
        try:
            while True:
                with self._cond:
                    while self._pending is None and not self._stop:
                        self._cond.wait()
                    if self._stop:
                        return
                    req, self._pending = self._pending, None
                try:
                    with self._lock:
                        cached = self._cache.get(req.path)
                    if cached is None:
                        self.detecting.emit(req.request)
                        if detector is None:
                            detector = StillDetector()
                        frame = first_frame(req.path)
                        found = detector.detect(frame)
                        h, w = frame.shape[:2]
                        dw, dh = fit_size(w, h, PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE)
                        s = dw / w
                        disp = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) if s < 1 else frame
                        cached = (disp, found.scaled(s))
                        with self._lock:
                            self._cache[req.path] = cached
                            while len(self._cache) > CACHE_FILES:
                                self._cache.popitem(last=False)
                    disp, found = cached
                    # The cache only fills after the detector exists, so it's always loaded here.
                    processor = MediaProcessor(detector, req.mode, req.effect, req.theme, req.hud, on_miss=req.on_miss)
                    out = processor.render_frame(disp, found)
                    self.ready.emit(req.request, bgr_to_qimage(out), str(req.path))
                except (RuntimeError, OSError, ValueError) as e:
                    self.failed.emit(req.request, str(e))
        finally:
            if detector is not None:
                detector.close()
