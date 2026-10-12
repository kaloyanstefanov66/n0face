import numpy as np

from conftest import H, W
from n0face.effects import make_effects
from n0face.mask import build_mask
from n0face.modes import MODES
from n0face.pipeline import anonymize, make_tracker

EDGES = np.random.default_rng(1).integers(0, 478, size=(800, 2))


def test_every_mode_effect_exists():
    effects = make_effects(EDGES)
    for mode in MODES.values():
        assert all(name in effects for name in mode.effects)


def test_only_mask_mode_renders_geometry():
    assert [m.name for m in MODES.values() if m.geometry] == ["mask"]


def test_anonymize_never_uses_camera_pixel_effects():
    effects = make_effects(EDGES)
    assert not any(effects[n].uses_camera_pixels for n in MODES["anonymize"].effects)


def test_ellipse_silhouette_hides_face_shape(face):
    """Different jaw shapes inside the same bounding box must give the same mask."""
    narrow_jaw = face.copy()
    center = face.mean(axis=0)
    lower = face[:, 1] > center[1] + 40
    narrow_jaw[lower, 0] = center[0] + (face[lower, 0] - center[0]) * 0.8
    kw = dict(margin=0.45, forehead=0.7, silhouette="ellipse")
    assert np.array_equal(build_mask([face], (H, W), **kw), build_mask([narrow_jaw], (H, W), **kw))


def test_anonymize_mode_covers_more_than_mask_mode(face):
    m, a = MODES["mask"], MODES["anonymize"]
    small = build_mask([face], (H, W), margin=m.margin, forehead=m.forehead, silhouette=m.silhouette)
    big = build_mask([face], (H, W), margin=a.margin, forehead=a.forehead, silhouette=a.silhouette)
    assert (big[small > 0] == 255).all() and big.sum() > small.sum()


def test_geometry_free_modes_ignore_landmark_positions(face):
    """In obfuscate/anonymize, moving landmarks inside the mask must not change the output."""
    frame = np.random.default_rng(3).integers(0, 256, (H, W, 3), dtype=np.uint8)
    effects = make_effects(EDGES)
    for name in ("obfuscate", "anonymize"):
        mode = MODES[name]
        jittered = face.copy()
        inner = np.linalg.norm(face - face.mean(axis=0), axis=1) < 30
        jittered[inner] += 5
        out1, _ = anonymize(frame, [face], make_tracker(mode), mode, effects[mode.effects[0]], 1.0)
        out2, _ = anonymize(frame, [jittered], make_tracker(mode), mode, effects[mode.effects[0]], 1.0)
        assert np.array_equal(out1, out2), name


def test_lost_face_hides_whole_frame_in_every_mode():
    frame = np.random.default_rng(4).integers(0, 256, (H, W, 3), dtype=np.uint8)
    effects = make_effects(EDGES)
    for mode in MODES.values():
        for effect_name in mode.effects:
            out, lost = anonymize(frame, [], make_tracker(mode), mode, effects[effect_name], 0.0)
            assert lost
            # Not a single row of the raw frame survives.
            assert not any(np.array_equal(out[y], frame[y]) for y in range(H)), (mode.name, effect_name)


def test_lost_face_never_uses_camera_pixels(face):
    """Even with pixelate selected, a lost face must hide the frame with input-independent content."""
    rng = np.random.default_rng(5)
    a, b = (rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(2))
    mode = MODES["obfuscate"]
    pixelate = make_effects(EDGES)["pixelate"]
    out_a, lost = anonymize(a, [], make_tracker(mode), mode, pixelate, 0.0)
    out_b, _ = anonymize(b, [], make_tracker(mode), mode, pixelate, 0.0)
    assert lost and np.array_equal(out_a, out_b)
