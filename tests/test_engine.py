import numpy as np

from conftest import H, W, FakeDetector
from n0face.pipeline import Engine
from n0face.theme import THEMES


def _frame():
    return np.random.default_rng(7).integers(0, 256, (H, W, 3), dtype=np.uint8)


def test_engine_hides_frame_until_face_locked():
    engine = Engine(FakeDetector([]))
    out = engine.process(_frame())
    assert engine.stats.lost and engine.stats.fail_closed_frames == 1
    assert not np.array_equal(out, _frame())


def test_engine_masks_face_and_reports_stats(face):
    engine = Engine(FakeDetector([face]), mode="obfuscate")
    frame = _frame()
    out = engine.process(frame)
    assert not engine.stats.lost and engine.stats.faces == 1
    assert np.array_equal(out[:, :40], frame[:, :40])  # background untouched
    cx, cy = np.round(face.mean(axis=0)).astype(int)
    assert not np.array_equal(out[cy, cx], frame[cy, cx])


def test_mode_switch_resets_to_fail_closed(face):
    detector = FakeDetector([face])
    engine = Engine(detector)
    engine.process(_frame())
    detector.faces = []
    engine.set_mode("anonymize")
    engine.process(_frame())
    assert engine.stats.lost


def test_effect_must_belong_to_mode():
    engine = Engine(FakeDetector([]), mode="anonymize")
    engine.set_effect("mesh")
    assert engine.effect_name == "glitch"


def test_theme_switch_keeps_session_tag():
    engine = Engine(FakeDetector([]))
    session = engine.hud.session
    engine.set_theme(THEMES["amber"])
    assert engine.hud.session == session and engine.effects["led"].theme == THEMES["amber"]
