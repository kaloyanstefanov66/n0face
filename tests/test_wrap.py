import numpy as np
import pytest

from n0face.detector import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not downloaded")
H, W = 480, 640


@pytest.fixture
def real_face():
    """Canonical face projected frontally into a 640x480 frame (478 landmarks)."""
    from n0face import canonical

    v = canonical.load().vertices
    pts = np.stack([320 + v[:, 0] * 12, 250 - v[:, 1] * 12], axis=1)
    return np.vstack([pts, pts[:10]]).astype(np.float32)  # pad the 10 iris points


@pytest.fixture
def cpu_wrap():
    from n0face import wrap

    old = (wrap.config.enabled, wrap.config.backend)
    wrap.config.enabled, wrap.config.backend = True, "cpu"
    yield
    wrap.config.enabled, wrap.config.backend = old


@pytest.mark.parametrize("name", ["mesh", "led", "circuit", "hex"])
def test_wrapped_mask_hugs_face_without_a_panel(name, real_face, cpu_wrap):
    from n0face.detector import _tesselation
    from n0face.effects import make_effects
    from n0face.mask import build_mask

    rng = np.random.default_rng(0)
    a, b = (rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(2))
    mask = build_mask([real_face], (H, W))
    effect = make_effects(_tesselation())[name]
    out_a = effect.apply(a, mask, [real_face], 1.0)
    out_b = effect.apply(b, mask, [real_face], 1.0)

    import cv2

    covered = (out_a != a).any(axis=2)
    assert covered.sum() > 5000
    # Inside the mask (all but its 1px antialiased rim) it's fully opaque: nothing of the
    # camera frame shows through.
    interior = cv2.erode(covered.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.array_equal(out_a[interior], out_b[interior])
    # No dark panel: just above the mesh (still inside the privacy hull) the frame is untouched.
    top = int(real_face[:, 1].min()) - 12
    assert mask[top, 320] == 255
    assert np.array_equal(out_a[top, 320], a[top, 320])


@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize("name", ["circuit", "hex"])
def test_preview_shows_textured_masks_not_a_black_panel(name, wrapped):
    """Regression: CIRCUIT/HEX rendered as a plain black panel in the 3D preview."""
    from n0face import wrap
    from n0face.app.preview3d import MaskScene, Pose
    from n0face.theme import DEFAULT_THEME

    old = (wrap.config.enabled, wrap.config.backend)
    wrap.config.enabled, wrap.config.backend = wrapped, "cpu"
    try:
        frame = MaskScene().render(320, 300, name, "mask", DEFAULT_THEME, 1.0, Pose(0.4, 0.1))
    finally:
        wrap.config.enabled, wrap.config.backend = old
    lit = (frame.image[..., :3].max(axis=2) > 150).sum()
    assert lit > 200  # glowing eye slits / traces / plates, not just a dark panel
