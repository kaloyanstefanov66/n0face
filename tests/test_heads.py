"""Heads from the pose model: covering people the face model misses, and dropping face hits
that land on a hand."""

import numpy as np
import pytest

from conftest import H, W
from n0face.detector import DEFAULT_MODEL
from n0face.heads import Person, match

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not downloaded")


@pytest.fixture(autouse=True)
def solid_backing():
    """Coverage here means 'no camera pixels', which is what the SOLID backing promises."""
    from n0face import wrap

    old = wrap.config.backing
    wrap.config.backing = "solid"
    yield
    wrap.config.backing = old


def test_head_ellipse_covers_the_head(head):
    cx, cy, ax, ay, angle = head.ellipse
    assert abs(angle) < 5
    assert all(head.contains(p) for p in head.points)
    assert ax > 80 and ay > ax  # wider than the eye-to-ear span, taller than wide
    assert head.contains((320, cy - 0.9 * ay))  # reaches the top of the head (hair)


def test_fitted_face_lands_on_the_head(head):
    from n0face.heads import _fit_face

    fit = _fit_face(head.points)
    for i, k in ((33, 6), (263, 3), (1, 0)):  # outer eye corners, nose: within a few pixels
        assert np.linalg.norm(fit[i] - head.points[k]) < 6
    lm = head.landmarks  # what gets drawn and covered: the fit, grown a little
    assert lm.shape == (478, 2)
    assert np.ptp(lm[:468, 0]) > np.ptp(fit[:468, 0])
    assert head.contains(lm[:468].mean(axis=0), 0.5)


def test_face_on_a_hand_is_dropped(head, face):
    from n0face.heads import _fit_face

    on_head = _fit_face(head.points)
    on_hand = face - face.mean(axis=0) + (520, 420)
    elsewhere = face - face.mean(axis=0) + (90, 90)  # maybe someone the pose model missed
    person = Person(head, hands=np.array([[530, 410], [500, 440]], np.float32))
    kept, heads = match([on_head, on_hand, elsewhere], [person])
    assert [id(f) for f in kept] == [id(on_head), id(elsewhere)]
    assert [h.has_face for h in heads] == [True]  # the person's face is drawn from its mesh, the head only covered
    kept, _ = match([on_hand], [person], drop_false=False)
    assert len(kept) == 1  # stale pose (live) never drops anything


def test_misplaced_face_mesh_gives_way_to_the_head(head):
    """Looking far down, the face model puts a frontal face on the hair above the real one."""
    from n0face.heads import _fit_face

    real = _fit_face(head.points)
    on_hair = real - (0, 0.45 * np.ptp(real[:468, 0]))  # still inside the head ellipse
    person = Person(head, np.zeros((0, 2), np.float32))
    kept, heads = match([real], [person])
    assert kept == [real] and [h.has_face for h in heads] == [True]
    kept, heads = match([on_hair], [person])
    assert kept == [] and [h.has_face for h in heads] == [False]  # drawn and covered from pose instead
    assert np.array_equal(heads[0].points, head.points)
    kept, heads = match([on_hair], [person], drop_false=False)  # live, pose a frame behind
    assert len(kept) == 1 and [h.has_face for h in heads] == [False]  # unsure: keep it, and cover the head too


def test_person_without_a_face_mesh_is_a_head(head, face):
    other = face - face.mean(axis=0) + (90, 90)
    kept, heads = match([other], [Person(head, np.zeros((0, 2), np.float32))])
    assert len(kept) == 1 and [h.has_face for h in heads] == [False]


@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize("effect_name", ["mesh", "x"])
def test_head_only_person_is_covered_in_mask_mode(head, effect_name, wrapped):
    """Someone the face model missed gets the design on a face fitted to their head. Even a
    3D-wrapped design covers that whole fitted face: no camera pixels show through it."""
    import cv2

    from n0face import wrap
    from n0face.detector import _tesselation
    from n0face.effects import make_effects
    from n0face.mask import build_mask
    from n0face.modes import MODES
    from n0face.pipeline import render_safe

    mode = MODES["mask"]
    rng = np.random.default_rng(0)
    a, b = (rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(2))
    mask = build_mask([], (H, W), mode.margin, mode.forehead, mode.silhouette, [head], head_fit=True)
    effect = make_effects(_tesselation())[effect_name]
    old = wrap.config.enabled
    wrap.config.enabled = wrapped
    try:
        out_a = render_safe(a, mask, [], False, mode, effect, 1.0, heads=[head])
        out_b = render_safe(b, mask, [], False, mode, effect, 1.0, heads=[head])
    finally:
        wrap.config.enabled = old
    must = build_mask([head.landmarks], (H, W), margin=0.0, forehead=0.0) if wrapped else mask
    inside = cv2.erode(must, np.ones((5, 5), np.uint8)) > 0  # all but the antialiased rim
    assert inside.sum() > 15000
    assert np.array_equal(out_a[inside], out_b[inside])


@pytest.mark.parametrize("mode_name", ["obfuscate", "anonymize"])
def test_head_only_person_is_fully_covered_in_privacy_modes(head, mode_name):
    """Obfuscate and anonymize don't rely on a fitted face: the whole head is covered."""
    from n0face.effects import make_effects
    from n0face.modes import MODES
    from n0face.pipeline import make_tracker, render_safe

    mode = MODES[mode_name]
    rng = np.random.default_rng(0)
    a, b = (rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(2))
    effect = make_effects(np.zeros((0, 2), int))["static"]
    outs = []
    for frame in (a, b):
        mask, faces, heads, lost = make_tracker(mode).update([], (H, W), [head])
        assert not lost
        outs.append(render_safe(frame, mask, faces, lost, mode, effect, 1.0, heads=heads))
    cx, cy, ax, ay, _ = head.ellipse
    ys, xs = np.mgrid[0:H, 0:W]
    whole_head = ((xs - cx) / (ax * 0.97)) ** 2 + ((ys - cy) / (ay * 0.97)) ** 2 <= 1
    assert np.array_equal(outs[0][whole_head], outs[1][whole_head])
