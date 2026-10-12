"""Frame sources: a webcam, or a video file (used for benchmarks)."""

import sys
import time

import cv2
import numpy as np

# A backend can open a camera yet deliver nothing (e.g. another app holds it). Each candidate
# must produce a real frame within this window before we commit to it.
WARMUP_S = 2.0


def _backends() -> tuple[tuple[str, int], ...]:
    if sys.platform == "win32":
        # Media Foundation gives full frame rate at 720p on many webcams where DirectShow is
        # capped at ~10 fps; DirectShow is the proven fallback.
        return (("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW))
    return (("ANY", cv2.CAP_ANY),)


class Camera:
    def __init__(self, source: int | str, width: int = 1280, height: int = 720, fps: int = 30):
        self.is_file = isinstance(source, str)
        self.backend = "FILE"
        self._first: np.ndarray | None = None
        if self.is_file:
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video source {source!r}")
            return

        try:  # failed grabs during warm-up are expected; keep OpenCV from flooding the log
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass
        for name, api in _backends():
            cap = cv2.VideoCapture(source, api)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            frame = self._warm_up(cap)
            if frame is not None:
                self._cap, self.backend, self._first = cap, name, frame
                return
            cap.release()
        raise RuntimeError(
            f"Could not get frames from camera {source}. Is another app using it, or is the "
            "camera blocked in Windows privacy settings?"
        )

    @staticmethod
    def _warm_up(cap) -> np.ndarray | None:
        deadline = time.perf_counter() + WARMUP_S
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.02)
        return None

    @property
    def size(self) -> tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> np.ndarray:
        """Return the next BGR frame. Webcam frames are mirrored like a selfie view."""
        if self._first is not None:
            frame, self._first = self._first, None
            return cv2.flip(frame, 1)
        ok, frame = self._cap.read()
        if not ok:
            if self.is_file:
                raise EOFError("End of video")
            # Tolerate a brief hiccup (USB glitch, exposure change) before giving up.
            for _ in range(10):
                time.sleep(0.02)
                ok, frame = self._cap.read()
                if ok:
                    break
            else:
                raise RuntimeError("Camera stopped delivering frames")
        return frame if self.is_file else cv2.flip(frame, 1)

    def close(self) -> None:
        self._cap.release()
