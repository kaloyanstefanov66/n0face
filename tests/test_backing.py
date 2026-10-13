"""Mask mode's blurred backing: the face under the design is blurred beyond recognition, and
fail-closed frames never get it."""

import cv2
import numpy as np
import pytest

from conftest import H, W
from n0face.detector import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not downloaded")


@pytest.fixture
def real_face():
    from n0face import canonical

    v = canonical.load().vertices
    pts = np.stack([320 + v[:, 0] * 12, 250 - v[:, 1] * 12], axis=1)
    return np.vstack([pts, pts[:10]]).astype(np.float32)


@pytest.fixture
def blur_on():
    from n0face import wrap

    old = (wrap.config.backing, wrap.config.enabled)
    wrap.config.backing = "blur"
    yield
    wrap.config.backing, wrap.config.enabled = old


def _detail(img, region):
    """High-frequency energy (Laplacian variance) inside `region`."""
    lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    return float(lap[region].var())


@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize("name", ["mesh", "x"])
def test_face_under_the_design_is_blurred(real_face, blur_on, name, wrapped):
    from n0face import wrap
    from n0face.detector import _tesselation
    from n0face.effects import make_effects
    from n0face.mask import build_mask
    from n0face.modes import MODES
    from n0face.pipeline import render_safe

    wrap.config.enabled = wrapped
    mode = MODES["mask"]
    frame = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)  # all detail
    mask = build_mask([real_face], (H, W), mode.margin, mode.forehead, mode.silhouette)
    out = render_safe(frame, mask, [real_face], False, mode, make_effects(_tesselation())[name], 1.0)
    face = build_mask([real_face], (H, W), margin=0.0, forehead=0.0) > 0
    # The design's own lines add some detail back; the camera's is gone.
    assert _detail(out, face) < 0.25 * _detail(frame, face)


def test_fail_closed_frame_is_never_blurred(blur_on):
    """A lost frame hides everything: a blur of the whole frame would still show the room."""
    from n0face.effects import make_effects
    from n0face.modes import MODES
    from n0face.pipeline import render_safe

    rng = np.random.default_rng(1)
    a, b = (rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(2))
    full = np.full((H, W), 255, np.uint8)
    effect = make_effects(np.zeros((0, 2), int))["mesh"]
    out_a = render_safe(a, full, [], True, MODES["mask"], effect, 1.0)
    out_b = render_safe(b, full, [], True, MODES["mask"], effect, 1.0)
    assert np.array_equal(out_a, out_b)
