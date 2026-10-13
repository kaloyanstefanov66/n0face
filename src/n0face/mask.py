"""Turn face landmarks into a privacy mask, and fail closed when tracking is lost.

The mask errs on the side of covering too much: it extends past the landmark hull to catch
hairline, ears and jaw, and it is OR-ed with the previous frame's mask so fast head
movement can't outrun it.
"""

import cv2
import numpy as np

ELLIPSE_QUANTUM_PX = 24


def face_polygon(landmarks: np.ndarray, margin: float = 0.15, forehead: float = 0.3) -> np.ndarray:
    """Expanded convex hull around one face's landmarks, as int32 polygon points.

    `margin` grows the hull uniformly around its centroid. `forehead` additionally stretches
    the upper half upward (as a fraction of face height), since the landmark mesh stops at
    mid-forehead.
    """
    hull = cv2.convexHull(landmarks.astype(np.float32)).reshape(-1, 2)
    center = hull.mean(axis=0)
    grown = center + (hull - center) * (1.0 + margin)

    top = grown[:, 1].min()
    height = grown[:, 1].max() - top
    upper = grown[:, 1] < center[1]
    # Pull upper points up proportionally to how high they already are, keeping the shape smooth.
    lift = (center[1] - grown[upper, 1]) / max(center[1] - top, 1e-6)
    grown[upper, 1] -= lift * forehead * height

    return cv2.convexHull(np.round(grown).astype(np.int32)).reshape(-1, 2)


def build_mask(
    faces: list[np.ndarray],
    shape: tuple[int, int],
    margin: float = 0.15,
    forehead: float = 0.3,
    silhouette: str = "hull",
    heads=(),
    head_fit: bool = False,
) -> np.ndarray:
    """uint8 mask (0 or 255) of the given (height, width) covering every face and head.

    `silhouette="hull"` follows the face outline. `"ellipse"` replaces it with an axis-aligned
    ellipse whose size is quantized, so the mask edge itself doesn't leak jaw or head shape.
    `heads` (n0face.heads.Head) the face model missed get a whole-head ellipse, or with
    `head_fit` (mask mode, hull silhouette) the same shape a face would, from the face mesh
    fitted to the head.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    q = ELLIPSE_QUANTUM_PX
    if head_fit and silhouette == "hull":
        faces, heads = [*faces, *(h.landmarks for h in heads if not h.has_face)], ()
    heads = [h for h in heads if not h.has_face]
    for head in heads:
        cx, cy, ax, ay, angle = head.ellipse
        ax, ay = ax * (1 + margin / 2), ay * (1 + margin / 2)
        if silhouette == "ellipse":
            # Axis-aligned box around the tilted ellipse, snapped to the grid like faces below.
            a = np.radians(angle)
            bx = np.hypot(ax * np.cos(a), ay * np.sin(a))
            by = np.hypot(ax * np.sin(a), ay * np.cos(a))
            center = (int(round(cx / (q / 2)) * (q / 2)), int(round(cy / (q / 2)) * (q / 2)))
            cv2.ellipse(mask, center, (int(np.ceil(bx / q) * q), int(np.ceil(by / q) * q)), 0, 0, 360, 255, -1)
        else:
            cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (int(np.ceil(ax)), int(np.ceil(ay))), angle, 0, 360, 255, -1)
    for landmarks in faces:
        if silhouette == "ellipse":
            # Only the landmark bounding box matters, and it's snapped to a grid, so the
            # mask doesn't follow jaw, cheek or chin shape.
            (x0, y0), (x1, y1) = landmarks.min(axis=0), landmarks.max(axis=0)
            w, h = x1 - x0, y1 - y0
            ax = w / 2 * (1 + margin)
            ay = h / 2 * (1 + margin) + forehead * h / 2
            center = (
                int(round((x0 + x1) / 2 / (q / 2)) * (q / 2)),
                int(round(((y0 + y1) / 2 - forehead * h / 2) / (q / 2)) * (q / 2)),
            )
            axes = (int(np.ceil(ax / q) * q), int(np.ceil(ay / q) * q))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        else:
            cv2.fillConvexPoly(mask, face_polygon(landmarks, margin, forehead), 255)
    return mask


class MaskTracker:
    """Smooths over brief detection dropouts, then fails closed.

    For up to `hold_frames` missed frames it keeps showing the last mask, growing it a little
    each frame in case the face is moving. After that, `lost` is True and the caller must hide
    the whole frame.
    """

    def __init__(self, hold_frames: int = 6, grow_px: int = 12, **mask_kwargs):
        self.hold_frames = hold_frames
        self.grow_px = grow_px
        self.mask_kwargs = mask_kwargs  # forwarded to build_mask
        self._prev_mask: np.ndarray | None = None
        self._prev_faces: list[np.ndarray] = []
        self._prev_heads: list = []
        self._misses = hold_frames + 1  # start "lost" until a face is seen

    def update(
        self, faces: list[np.ndarray], shape: tuple[int, int], heads=()
    ) -> tuple[np.ndarray, list[np.ndarray], list, bool]:
        """Returns (mask, faces to render, heads to render, lost)."""
        if faces or heads:
            current = build_mask(faces, shape, heads=heads, **self.mask_kwargs)
            mask = current.copy()
            if self._prev_mask is not None and self._prev_mask.shape == mask.shape:
                mask |= self._prev_mask
            # Remember only this frame's coverage, so the union doesn't accumulate forever.
            self._prev_mask = current
            self._prev_faces, self._prev_heads = faces, list(heads)
            self._misses = 0
            return mask, faces, list(heads), False

        self._misses += 1
        if self._misses > self.hold_frames or self._prev_mask is None:
            return np.full(shape, 255, dtype=np.uint8), [], [], True

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.grow_px + 1,) * 2)
        self._prev_mask = cv2.dilate(self._prev_mask, kernel)
        return self._prev_mask.copy(), self._prev_faces, self._prev_heads, False
