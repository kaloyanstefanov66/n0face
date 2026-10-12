"""Face landmark detection with MediaPipe FaceLandmarker, run entirely on-device."""

from pathlib import Path

import cv2
import numpy as np

from n0face.resources import resource

DEFAULT_MODEL = resource("models", "face_landmarker.task")


def _tesselation() -> np.ndarray:
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

    return np.array(
        [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION],
        dtype=np.int32,
    )


def _create_landmarker(model_path: Path, max_faces: int, video: bool):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run: python scripts/get_model.py")
    landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO if video else RunningMode.IMAGE,
            num_faces=max_faces,
            # Low thresholds: a missed face leaks, a false positive just masks a bit of wall.
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )
    return mp, landmarker


def _to_pixels(result, w: int, h: int, offset=(0, 0)) -> list[np.ndarray]:
    ox, oy = offset
    return [
        np.array([(p.x * w + ox, p.y * h + oy) for p in face], dtype=np.float32)
        for face in result.face_landmarks
    ]


class FaceDetector:
    """Live video detector with temporal tracking. Returns one (478, 2) pixel array per face."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, max_faces: int = 4):
        self._mp, self._landmarker = _create_landmarker(model_path, max_faces, video=True)
        self.edges = _tesselation()
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        h, w = frame_bgr.shape[:2]
        return _to_pixels(result, w, h)

    def close(self) -> None:
        self._landmarker.close()


def _bbox(lm: np.ndarray) -> tuple[float, float, float, float]:
    (x0, y0), (x1, y1) = lm.min(axis=0), lm.max(axis=0)
    return x0, y0, x1, y1


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def tile_grid(w: int, h: int, min_tile: int = 512, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Overlapping square tiles (x, y, tw, th) covering the image. Empty for small images."""
    side = max(min_tile, min(w, h) // 2)
    if max(w, h) <= side:
        return []
    step = max(int(side * (1 - overlap)), 1)
    xs = list(range(0, max(w - side, 0) + 1, step))
    ys = list(range(0, max(h - side, 0) + 1, step))
    if xs[-1] + side < w:
        xs.append(w - side)
    if ys[-1] + side < h:
        ys.append(h - side)
    return [(x, y, min(side, w - x), min(side, h - y)) for y in ys for x in xs]


class StillDetector:
    """Detector for photos and recorded video frames, with no tracking between calls.

    With `thorough=True` it also scans overlapping tiles, which finds small faces (group photos,
    people in the background) that a single full-image pass misses.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL, max_faces: int = 20, thorough: bool = True):
        self._mp, self._landmarker = _create_landmarker(model_path, max_faces, video=False)
        self.edges = _tesselation()
        self.thorough = thorough

    def _detect_region(self, rgb: np.ndarray, x: int, y: int) -> list[np.ndarray]:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        h, w = rgb.shape[:2]
        return _to_pixels(self._landmarker.detect(image), w, h, (x, y))

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0, tiles: bool | None = None) -> list[np.ndarray]:
        """`tiles` overrides `thorough` for this call (videos tile only every few frames)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self._detect_region(rgb, 0, 0)
        if self.thorough if tiles is None else tiles:
            h, w = rgb.shape[:2]
            for x, y, tw, th in tile_grid(w, h):
                for lm in self._detect_region(rgb[y : y + th, x : x + tw], x, y):
                    if all(_iou(_bbox(lm), _bbox(f)) < 0.3 for f in faces):
                        faces.append(lm)
        return faces

    def close(self) -> None:
        self._landmarker.close()
