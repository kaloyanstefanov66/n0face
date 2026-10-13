"""Finding heads with MediaPipe's pose model, for when the face model can't.

The face model misses heads tilted far down or away (looking at a guitar, a keyboard, a
phone) and now and then fires on a hand. The pose model tracks the whole person, so it
still knows where the head is. Detectors use it two ways:

- a person whose head no face mesh matches gets a head-sized mask (`Head`), so looking
  down never uncovers the face and never blanks the frame;
- a "face" that sits on a person's hand instead of any head is dropped as a false hit.

Like the face model, it runs on-device from a bundled file.
"""

import math
import threading
from dataclasses import dataclass
from functools import cached_property, lru_cache

import cv2
import numpy as np

from n0face.resources import resource

POSE_MODEL = resource("models", "pose_landmarker.task")
HAND_POINTS = list(range(15, 23))  # wrists, pinkies, index fingers, thumbs
MATCH_SCALE = 1.3  # a face belongs to a head if its center is inside the head ellipse grown by this
AGREE = 0.2  # ...and if its nose and eyes are within this many face widths of the pose keypoints
FIT_GROW = 1.12  # a face mesh fitted to pose keypoints is drawn this much larger, to absorb fit error


@dataclass(frozen=True, eq=False)
class Head:
    """A person's head, from the pose model's 11 head keypoints (pixels).

    Keypoint order: nose, left eye inner/center/outer, right eye inner/center/outer, left ear,
    right ear, mouth left, mouth right ("left" is the person's left).
    """

    points: np.ndarray  # (11, 2)
    mesh: np.ndarray | None = None  # face mesh to draw, if already known (e.g. smoothed over time)
    has_face: bool = False  # the face model found this person's face: cover only, don't draw

    @cached_property
    def ellipse(self) -> tuple[float, float, float, float, float]:
        """(cx, cy, ax, ay, angle_deg): covers the whole head, hair included."""
        k = self.points
        ears, eyes, mouth = k[[7, 8]], k[[2, 5]], k[[9, 10]]
        eye_line = eyes[1] - eyes[0]
        size = max(
            float(np.linalg.norm(ears[0] - ears[1])),
            2.4 * float(np.linalg.norm(eyes.mean(axis=0) - mouth.mean(axis=0))),
            1.8 * float(np.linalg.norm(k[3] - k[6])),
            1.0,
        )
        angle = math.degrees(math.atan2(eye_line[1], eye_line[0]))
        if abs(angle) > 90:  # eyes listed right-to-left in the image: same line, other way round
            angle -= math.copysign(180, angle)
        up = np.array([math.sin(math.radians(angle)), -math.cos(math.radians(angle))])
        center = (ears.mean(axis=0) + eyes.mean(axis=0)) / 2 + up * 0.15 * size
        return float(center[0]), float(center[1]), 0.85 * size, 1.1 * size, angle

    def contains(self, p, grow: float = 1.0) -> bool:
        cx, cy, ax, ay, angle = self.ellipse
        a = math.radians(angle)
        dx, dy = p[0] - cx, p[1] - cy
        u = (dx * math.cos(a) + dy * math.sin(a)) / (ax * grow)
        v = (-dx * math.sin(a) + dy * math.cos(a)) / (ay * grow)
        return u * u + v * v <= 1

    @cached_property
    def landmarks(self) -> np.ndarray:
        """A (478, 2) face mesh fitted to the head keypoints (grown by FIT_GROW), so mask designs
        can be drawn on it. It tracks the face well even looking far down."""
        if self.mesh is not None:
            return self.mesh
        lm = _fit_face(self.points)
        center = lm[:468].mean(axis=0)
        return (center + (lm - center) * FIT_GROW).astype(np.float32)

    def crop(self, w: int, h: int, scale: float = 1.6) -> tuple[int, int, int, int] | None:
        """A square region (x0, y0, x1, y1) around the head, for a closer look with the face model."""
        cx, cy, _, ay, _ = self.ellipse
        half = scale * ay
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
        return (x0, y0, x1, y1) if x1 - x0 > 16 and y1 - y0 > 16 else None

    def scaled(self, s: float) -> "Head":
        mesh = None if self.mesh is None else self.mesh * np.float32(s)
        return Head(self.points * np.float32(s), mesh, self.has_face)

    def agrees(self, lm: np.ndarray, tolerance: float = AGREE) -> bool:
        """Whether a face mesh's nose and eyes are where the pose model puts them. Looking far
        down, the face model tends to place a frontal face on the hair above the real one."""
        width = max(float(np.ptp(lm[:468, 0])), 1.0)
        nose = np.linalg.norm(lm[1] - self.points[0])
        eyes = np.linalg.norm((lm[33] + lm[263]) / 2 - (self.points[2] + self.points[5]) / 2)
        return max(nose, eyes) < tolerance * width


@dataclass(frozen=True)
class Person:
    head: Head
    hands: np.ndarray  # (k, 2) visible hand keypoints


# ---- face mesh from pose keypoints ---------------------------------------------------------

# Canonical face mesh points for each pose head keypoint (a pair means their midpoint). Both
# use the person's own left and right (mesh 263 is the left eye's outer corner), so fitted
# meshes always have the same handedness and can be averaged over time.
_MESH_POINTS = [(1,), (362,), (362, 263), (263,), (133,), (133, 33), (33,), (454,), (234,), (291,), (61,)]


@lru_cache(maxsize=1)
def _canonical():
    from n0face import canonical

    v = canonical.load().vertices.astype(np.float64) * np.array([1, -1, 1])  # y down, like pixels
    return v, np.array([v[list(ix)].mean(axis=0) for ix in _MESH_POINTS])


def _fit_face(points: np.ndarray) -> np.ndarray:
    v, ref = _canonical()
    p = points.astype(np.float64)
    rc, pc = ref - ref.mean(axis=0), p - p.mean(axis=0)
    a, *_ = np.linalg.lstsq(rc, pc, rcond=None)  # (3, 2): weak perspective, then made rigid
    s = (np.linalg.norm(a[:, 0]) + np.linalg.norm(a[:, 1])) / 2
    r0 = a[:, 0] / max(np.linalg.norm(a[:, 0]), 1e-9)
    r1 = a[:, 1] - (a[:, 1] @ r0) * r0
    r1 /= max(np.linalg.norm(r1), 1e-9)
    mesh = ((v - ref.mean(axis=0)) @ (s * np.stack([r0, r1], axis=1)) + p.mean(axis=0)).astype(np.float32)
    return np.vstack([mesh, mesh[:10]])  # 468 mesh points + 10 stand-ins for the iris points


# ---- matching faces to people --------------------------------------------------------------


def match(faces: list[np.ndarray], people: list[Person], drop_false: bool = True) -> tuple[list[np.ndarray], list[Head]]:
    """Faces to keep, and every person's head (`has_face` set when one of the kept faces is theirs).

    A face whose center is inside a head is that person's face, if it agrees with the pose
    keypoints; if not, it's misplaced and the head is drawn from pose instead. A face that is in
    no head but sits on someone's hand is a false hit and is dropped; any other face is kept,
    since it may be a person the pose model didn't see. `drop_false=False` (pose results that
    trail the frame) drops nothing and allows more disagreement.
    """
    kept, covered = [], set()
    tolerance = AGREE if drop_false else 2 * AGREE
    for lm in faces:
        center = lm[:468].mean(axis=0)
        owner = next((i for i, p in enumerate(people) if p.head.contains(center, MATCH_SCALE)), None)
        if owner is not None:
            if people[owner].head.agrees(lm, tolerance):
                covered.add(owner)
                kept.append(lm)
            elif not drop_false:
                kept.append(lm)  # can't tell yet: keep it, and the head gets covered too
            continue
        width = float(np.ptp(lm[:468, 0]))
        on_hand = any(
            len(p.hands) and float(np.min(np.linalg.norm(p.hands - center, axis=1))) < 1.2 * width for p in people
        )
        if not (drop_false and on_hand):
            kept.append(lm)
    return kept, [Head(p.head.points, has_face=i in covered) for i, p in enumerate(people)]


# ---- the model -----------------------------------------------------------------------------


class PoseFinder:
    """People in a frame, from MediaPipe PoseLandmarker."""

    def __init__(self, video: bool, max_people: int = 4, model_path=POSE_MODEL):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run: python scripts/get_model.py")
        self._mp = mp
        self._video = video
        self._last_ts = -1
        self._model = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO if video else RunningMode.IMAGE,
                num_poses=max_people,
                # Low thresholds, like the face model: a missed head can leak.
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )

    def find(self, rgb: np.ndarray, timestamp_ms: int = 0) -> list[Person]:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video:
            timestamp_ms = max(timestamp_ms, self._last_ts + 1)
            self._last_ts = timestamp_ms
            result = self._model.detect_for_video(image, timestamp_ms)
        else:
            result = self._model.detect(image)
        h, w = rgb.shape[:2]
        people = []
        for lm in result.pose_landmarks:
            pts = np.array([(p.x * w, p.y * h) for p in lm], np.float32)
            vis = np.array([p.visibility for p in lm])
            hands = pts[HAND_POINTS][vis[HAND_POINTS] > 0.3]
            people.append(Person(Head(pts[:11]), hands))
        return people

    def close(self) -> None:
        self._model.close()


class BackgroundPose:
    """Runs a video PoseFinder on its own thread, for the live camera.

    `submit` hands over the newest frame (dropping one still waiting) and `latest` returns the
    most recent result without waiting, so pose never slows the frame loop. Results trail
    the camera by a frame or two; `fresh` says whether they belong to the last submitted frame.
    """

    def __init__(self, max_people: int = 4):
        self._finder = PoseFinder(video=True, max_people=max_people)
        self._cond = threading.Condition()
        self._job: tuple[np.ndarray, int, int] | None = None
        self._result: tuple[list[Person], int] = ([], -1)
        self._seq = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="pose", daemon=True)
        self._thread.start()

    def submit(self, rgb: np.ndarray, timestamp_ms: int) -> None:
        with self._cond:
            self._seq += 1
            self._job = (rgb, timestamp_ms, self._seq)
            self._cond.notify()

    def latest(self) -> tuple[list[Person], bool]:
        with self._cond:
            people, seq = self._result
            return people, seq == self._seq

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._job is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                rgb, ts, seq = self._job
                self._job = None
            people = self._finder.find(rgb, ts)
            with self._cond:
                self._result = (people, seq)

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=2)
        self._finder.close()
