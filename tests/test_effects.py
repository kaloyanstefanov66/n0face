import numpy as np
import pytest

from conftest import H, W
from n0face.effects import EFFECT_NAMES, make_effects
from n0face.mask import build_mask

EDGES = np.random.default_rng(1).integers(0, 478, size=(800, 2))


@pytest.fixture(params=EFFECT_NAMES)
def effect(request):
    return make_effects(EDGES)[request.param]


def _frames():
    rng = np.random.default_rng(2)
    return (
        rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
        rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
    )


def test_masked_region_is_independent_of_camera_pixels(effect, face):
    """Two completely different camera frames must produce identical pixels inside the mask."""
    if effect.uses_camera_pixels:
        pytest.skip("derived from camera pixels by design; documented as weakest")
    a, b = _frames()
    mask = build_mask([face], (H, W))
    out_a = effect.apply(a, mask, [face], t=1.25)
    out_b = effect.apply(b, mask, [face], t=1.25)
    inside = mask > 0
    assert np.array_equal(out_a[inside], out_b[inside])


def test_pixels_far_outside_mask_are_untouched(effect, face):
    a, _ = _frames()
    mask = build_mask([face], (H, W))
    out = effect.apply(a, mask, [face], t=0.5)
    assert np.array_equal(out[:, :40], a[:, :40])


def test_full_frame_obscure_without_faces(effect):
    if effect.uses_camera_pixels:
        pytest.skip("derived from camera pixels by design")
    a, b = _frames()
    full = np.full((H, W), 255, dtype=np.uint8)
    assert np.array_equal(effect.apply(a, full, [], t=2.0), effect.apply(b, full, [], t=2.0))


def test_output_shape_and_dtype(effect, face):
    a, _ = _frames()
    out = effect.apply(a, build_mask([face], (H, W)), [face], t=0.0)
    assert out.shape == a.shape and out.dtype == np.uint8


def test_only_pixelate_uses_camera_pixels():
    assert [n for n, e in make_effects(EDGES).items() if e.uses_camera_pixels] == ["pixelate"]


def test_pixelate_changes_the_masked_region(face):
    a, _ = _frames()
    mask = build_mask([face], (H, W))
    out = make_effects(EDGES)["pixelate"].apply(a, mask, [face], t=0.0)
    inside = mask > 0
    assert (out[inside] != a[inside]).mean() > 0.9
