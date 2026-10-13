"""Face landmark detection with MediaPipe FaceLandmarker, run entirely on-device.

Each detector also runs the pose model (n0face.heads): it drops face hits that land on a
hand, and reports the heads of people whose face the face model missed.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from n0face.heads import BackgroundPose, Head, PoseFinder, match
from n0face.resources import resource

DEFAULT_MODEL = resource("models", "face_landmarker.task")
CROP_SIDE = 384  # head crops are resized to this before the face model looks again


@dataclass
class Found:
    """What a detector saw in one frame."""

    faces: list[np.ndarray] = field(default_factory=list)  # (478, 2) face meshes, pixels
    heads: list[Head] = field(default_factory=list)  # people whose face has no mesh

    def __bool__(self) -> bool:
        return bool(self.faces or self.heads)

    def scaled(self, s: float) -> "Found":
        return Found([f * s for f in self.faces], [h.scaled(s) for h in self.heads])

    @staticmethod
    def union(found: "list[Found]") -> "Found":
        return Found([f for x in found for f in x.faces], [h for x in found for h in x.heads])


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
    """Live video detector with temporal tracking.

    Pose runs on a background thread so it never slows the frame loop; its result trails by a
    frame or two, so face hits are only dropped as false when it's current.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL, max_faces: int = 4, pose: bool = True):
        self._mp, self._landmarker = _create_landmarker(model_path, max_faces, video=True)
        self._pose = BackgroundPose(max_faces) if pose else None
        self.edges = _tesselation()
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Found:
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._pose is not None:
            self._pose.submit(rgb, timestamp_ms)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        h, w = frame_bgr.shape[:2]
        faces = _to_pixels(result, w, h)
        if self._pose is None:
            return Found(faces)
        people, fresh = self._pose.latest()
        return Found(*match(faces, people, drop_false=fresh))

    def close(self) -> None:
        self._landmarker.close()
        if self._pose is not None:
            self._pose.close()


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

    def __init__(self, model_path: Path = DEFAULT_MODEL, max_faces: int = 20, thorough: bool = True, pose: bool = True):
        self._mp, self._landmarker = _create_landmarker(model_path, max_faces, video=False)
        self._pose = PoseFinder(video=False, max_people=6) if pose else None
        self.edges = _tesselation()
        self.thorough = thorough

    def _detect_region(self, rgb: np.ndarray, x: int, y: int) -> list[np.ndarray]:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        h, w = rgb.shape[:2]
        return _to_pixels(self._landmarker.detect(image), w, h, (x, y))

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0, tiles: bool | None = None) -> Found:
        """`tiles` overrides `thorough` for this call (videos tile only every few frames)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self._detect_region(rgb, 0, 0)
        if self.thorough if tiles is None else tiles:
            h, w = rgb.shape[:2]
            for x, y, tw, th in tile_grid(w, h):
                for lm in self._detect_region(rgb[y : y + th, x : x + tw], x, y):
                    if all(_iou(_bbox(lm), _bbox(f)) < 0.3 for f in faces):
                        faces.append(lm)
        if self._pose is None:
            return Found(faces)
        people = self._pose.find(rgb)
        # A closer look around each head no face covers yet: small faces that the full-image
        # pass misses are found at a fraction of the cost of the tiles.
        h, w = rgb.shape[:2]
        for person in people:
            box = person.head.crop(w, h)
            if box is None or any(person.head.contains(f[:468].mean(axis=0), 1.3) for f in faces):
                continue
            x0, y0, x1, y1 = box
            s = CROP_SIDE / max(x1 - x0, y1 - y0)
            crop = cv2.resize(rgb[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            for lm in self._detect_region(crop, 0, 0):
                lm = lm / s + (x0, y0)
                if all(_iou(_bbox(lm), _bbox(f)) < 0.3 for f in faces):
                    faces.append(lm.astype(np.float32))
        return Found(*match(faces, people))

    def close(self) -> None:
        self._landmarker.close()
        if self._pose is not None:
            self._pose.close()
