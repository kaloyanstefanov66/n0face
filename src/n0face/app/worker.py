"""Capture/processing threads. The UI never touches the camera or files directly.

Frames sent to the UI are scaled down here to the size they're displayed at, so the UI thread
never has to convert or scale full-resolution frames.
"""

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from n0face.app.services import bgr_to_qimage, fit_size

from n0face.capture import Camera
from n0face.output import VirtualCam, VirtualCamUnavailable
from n0face.pipeline import Engine, Stats


class Worker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(object)  # Stats
    camera_state = Signal(bool)
    stream_state = Signal(bool, str)  # on, device name or error
    failed = Signal(str)

    def __init__(self, engine: Engine, camera_index: int, stream: bool, size=(1280, 720), fps: int = 30):
        super().__init__()
        self.engine = engine
        self.camera_index = camera_index
        self.want_stream = stream
        self.size = size
        self.fps = fps
        self._stop = False
        self.display_size: tuple[int, int] | None = None  # set by the UI on resize

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            camera = Camera(self.camera_index, *self.size, self.fps)
        except RuntimeError as e:
            self.failed.emit(str(e))
            return
        self.camera_state.emit(True)
        # Start every session fail-closed: nothing is published until a face is locked.
        self.engine.set_mode(self.engine.mode_name, self.engine.effect_name)

        sink = None
        n = 0
        try:
            while not self._stop:
                if self.want_stream and sink is None:
                    sink = self._open_sink(camera)
                elif not self.want_stream and sink is not None:
                    sink.close()
                    sink = None
                    self.stream_state.emit(False, "")

                try:
                    frame = camera.read()
                except RuntimeError as e:
                    self.failed.emit(str(e))
                    break
                safe = self.engine.process(frame)
                if sink is not None:
                    sink.send(safe)

                self.frame_ready.emit(bgr_to_qimage(downscale(safe, self.display_size)))
                n += 1
                if n % 6 == 0:
                    s = self.engine.stats
                    self.stats_ready.emit(Stats(**vars(s)))
        finally:
            camera.close()
            if sink is not None:
                sink.close()
                self.stream_state.emit(False, "")
            self.camera_state.emit(False)

    def _open_sink(self, camera: Camera):
        try:
            sink = VirtualCam(*camera.size, self.fps)
        except VirtualCamUnavailable as e:
            self.want_stream = False
            self.stream_state.emit(False, str(e))
            return None
        self.stream_state.emit(True, sink.device)
        return sink


class FileWorker(QThread):
    """Processes photos/videos one after another with the offline (look-ahead) pipeline."""

    frame_ready = Signal(QImage)
    progress = Signal(int, int, float, str)  # file index (1-based), file count, fraction, name
    file_done = Signal(object)  # media.Report
    file_failed = Signal(str, str)  # name, error
    all_done = Signal(int, int, bool)  # ok, failed, cancelled

    def __init__(self, files, out_dir, mode, effect, theme, hud, on_miss, keep_audio):
        super().__init__()
        self.files = files
        self.out_dir = out_dir
        self.options = dict(mode=mode, effect=effect, theme=theme, hud=hud, on_miss=on_miss, keep_audio=keep_audio)
        self._stop = False
        self._current = (0, "")
        self.display_size: tuple[int, int] | None = None

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        from n0face.detector import StillDetector
        from n0face.media import Cancelled, MediaProcessor

        detector = StillDetector()
        processor = MediaProcessor(detector, progress=self._on_progress, cancelled=lambda: self._stop, **self.options)
        ok = failed = 0
        cancelled = False
        try:
            for i, src in enumerate(self.files, 1):
                if self._stop:
                    cancelled = True
                    break
                self._current = (i, src.name)
                self.progress.emit(i, len(self.files), 0.0, src.name)
                try:
                    self.file_done.emit(processor.process(src, self.out_dir))
                    ok += 1
                except Cancelled:
                    cancelled = True
                    break
                except (RuntimeError, OSError, ValueError) as e:
                    failed += 1
                    self.file_failed.emit(src.name, str(e))
        finally:
            detector.close()
            self.all_done.emit(ok, failed, cancelled)

    def _on_progress(self, frac: float, frame) -> None:
        i, name = self._current
        self.progress.emit(i, len(self.files), frac, name)
        if frame is not None:
            self.frame_ready.emit(bgr_to_qimage(downscale(frame, self.display_size)))


def downscale(img: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if size is None:
        return img
    h, w = img.shape[:2]
    tw, th = fit_size(w, h, *size)
    if tw >= w:
        return img
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
