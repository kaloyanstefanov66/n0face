import numpy as np

from conftest import H, W
from n0face.mask import MaskTracker, build_mask


def test_mask_covers_every_landmark(face):
    mask = build_mask([face], (H, W))
    xs, ys = np.round(face).astype(int).T
    assert (mask[ys, xs] == 255).all()


def test_mask_extends_above_landmarks_for_forehead(face):
    mask = build_mask([face], (H, W))
    top_landmark = int(face[:, 1].min())
    assert mask[: top_landmark - 20, 300:340].any()


def test_tracker_starts_lost_and_hides_whole_frame():
    mask, faces, _, lost = MaskTracker().update([], (H, W))
    assert lost and faces == []
    assert (mask == 255).all()


def test_tracker_holds_then_fails_closed(face):
    tracker = MaskTracker(hold_frames=3)
    first, _, _, lost = tracker.update([face], (H, W))
    assert not lost

    for _ in range(3):
        held, faces, _, lost = tracker.update([], (H, W))
        assert not lost and len(faces) == 1
        # Held masks only grow, so a moving face stays covered.
        assert (held[first > 0] == 255).all()

    mask, faces, _, lost = tracker.update([], (H, W))
    assert lost and faces == []
    assert (mask == 255).all()


def test_tracker_unions_with_previous_frame(face):
    tracker = MaskTracker()
    tracker.update([face], (H, W))
    moved = face + np.array([150, 0], dtype=np.float32)
    mask, _, _, _ = tracker.update([moved], (H, W))
    assert (mask[build_mask([face], (H, W)) > 0] == 255).all()
    assert (mask[build_mask([moved], (H, W)) > 0] == 255).all()
