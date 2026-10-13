import numpy as np
import pytest

from n0face import glitch
from n0face.theme import DEFAULT_THEME


@pytest.fixture
def glitch_on():
    old = glitch.config.enabled
    glitch.config.enabled = True
    yield
    glitch.config.enabled = old


def burst_time() -> float:
    ts = np.arange(0, 30, 1 / 30)
    return float(ts[np.argmax([glitch.strength(t) for t in ts])])


def test_bursts_are_occasional():
    s = np.array([glitch.strength(t) for t in np.arange(0, 120, 1 / 30)])
    assert 0.03 < (s > 0).mean() < 0.3  # mostly clean frames
    assert (s > 0.5).any()


@pytest.mark.parametrize("channels", [3, 4])
def test_only_touches_the_region_around_the_face(channels):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (240, 320, channels), dtype=np.uint8)
    before = img.copy()
    glitch.apply(img, (130, 80, 190, 160), burst_time(), DEFAULT_THEME)  # face 60 px wide
    assert not np.array_equal(img, before)
    changed = (img != before).any(axis=2)
    ys, xs = np.nonzero(changed)
    assert ys.min() >= 80 and ys.max() < 160
    assert xs.min() >= 130 - 21 and xs.max() < 190 + 21  # tears may run 35% of the width past it


def test_no_glitch_outside_mask_mode(glitch_on):
    """Obfuscate/anonymize must stay exactly as without the option (it's a mask-mode look)."""
    from n0face.effects import make_effects
    from n0face.modes import MODES
    from n0face.pipeline import render_safe

    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), np.uint8)
    mask[30:90, 50:110] = 255
    effect = make_effects(np.zeros((0, 2), int))["static"]
    t = burst_time()
    out_on = render_safe(frame, mask, [], False, MODES["anonymize"], effect, t)
    glitch.config.enabled = False
    out_off = render_safe(frame, mask, [], False, MODES["anonymize"], effect, t)
    assert np.array_equal(out_on, out_off)


@pytest.fixture
def knobs():
    old = (glitch.config.intensity, glitch.config.frequency, glitch.config.length)
    yield glitch.config
    glitch.config.intensity, glitch.config.frequency, glitch.config.length = old


def _glitched_share(seconds=120):
    return np.mean([glitch.strength(t) > 0 for t in np.arange(0, seconds, 1 / 30)])


def test_frequency_and_length_change_how_much_is_glitched(knobs):
    base = _glitched_share()
    knobs.frequency = 3.0
    often = _glitched_share()
    knobs.frequency, knobs.length = 1.0, 2.0
    longer = _glitched_share()
    knobs.length, knobs.frequency = 1.0, 0.25
    rare = _glitched_share()
    assert rare < base < often and longer > base


def test_intensity_scales_the_tear(knobs):
    t = burst_time()
    peak = glitch.strength(t)
    knobs.intensity = 2.0
    assert glitch.strength(t) == pytest.approx(2 * peak)
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    hard = img.copy()
    glitch.apply(hard, (130, 80, 190, 160), t, DEFAULT_THEME)
    knobs.intensity = 0.25
    soft = img.copy()
    glitch.apply(soft, (130, 80, 190, 160), t, DEFAULT_THEME)
    assert (hard != img).any(axis=2).sum() > (soft != img).any(axis=2).sum()
