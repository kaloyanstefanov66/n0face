import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module", autouse=True)
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def run(spring, seconds):
    t = 0.0
    while t < seconds:
        if spring._step(1 / 60):
            break
        t += 1 / 60
    return spring


def test_spring_settles_on_target():
    from n0face.app.motion import SMOOTH, Spring

    s = Spring(0, SMOOTH)
    s.target = 1.0
    run(s, 3)
    assert s.value == 1.0 and s.velocity == 0.0


def test_bouncy_spring_overshoots_smooth_does_not():
    from n0face.app.motion import BOUNCY, GENTLE, Spring

    def peak(preset):
        s = Spring(0, preset)
        s.target = 1.0
        best = 0.0
        for _ in range(240):
            s._step(1 / 120)
            best = max(best, s.value)
        return best

    assert peak(BOUNCY) > 1.02
    assert peak(GENTLE) <= 1.0 + 1e-6


def test_reduced_motion_snaps():
    from n0face.app import motion
    from n0face.app.motion import SMOOTH, Spring

    motion.set_reduced_motion(True)
    try:
        s = Spring(0, SMOOTH)
        s.to(1)
        assert s.value == 1.0 and not s.moving
    finally:
        motion.set_reduced_motion(False)


def test_orbit_fling_decays_and_hands_back_to_auto_sway():
    from n0face.app.viewport import Orbit

    o = Orbit()
    o.auto.snap(1)
    o.start_drag(t=2.0)
    assert o.auto.value == 0
    o.drag(30, 0, 1 / 60)
    o.end_drag(stale=False)
    assert abs(o.vyaw) > 0
    assert abs(o.vyaw) <= 6.0  # flings are capped
    yaw0 = o.yaw
    for _ in range(20):
        o.step(1 / 60)
    assert o.yaw > yaw0  # inertia carried it on
    for _ in range(60 * 5):
        o.step(1 / 60)
    assert abs(o.vyaw) < 0.02  # friction stopped it
    assert o.auto.target == 1  # idle long enough: auto-sway resumes (blended)
    assert -math.pi <= o.yaw <= math.pi


def test_orbit_stale_release_does_not_fling():
    from n0face.app.viewport import Orbit

    o = Orbit()
    o.start_drag(t=0.0)
    o.drag(40, 0, 1 / 60)
    o.end_drag(stale=True)
    assert o.vyaw == 0 and o.vpitch == 0


def test_orbit_reset_returns_home():
    from n0face.app.viewport import Orbit

    o = Orbit()
    o.auto_enabled = False
    o.start_drag(t=0.0)
    o.drag(150, 60, 0.5)
    o.end_drag(stale=True)
    o.reset()
    for _ in range(60 * 3):
        o.step(1 / 60)
    assert abs(o.yaw) < 1e-2 and abs(o.pitch) < 1e-2


def _process_deletes():
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_clock_drops_springs_of_deleted_widgets():
    """Regression: deleting a widget mid-animation crashed the clock ("already deleted")."""
    from n0face.app.motion import clock
    from n0face.app.widgets import ChoiceRow

    row = ChoiceRow(1, "MESH")
    row.show()
    row.hover.to(1)  # mid-animation...
    assert row.hover in clock().springs
    spring = row.hover
    row.deleteLater()  # ...when the row is deleted
    _process_deletes()
    assert not spring.alive()
    clock()._tick()  # must not raise
    assert spring not in clock().springs


def test_switching_modes_while_effect_rows_animate(monkeypatch):
    """Regression for the obfuscate crash: hover an effect row, then switch mode."""
    from PySide6.QtCore import QSettings

    from n0face.app import window as app_window
    from n0face.app.motion import clock
    from n0face.detector import DEFAULT_MODEL

    if not DEFAULT_MODEL.exists():
        pytest.skip("model not downloaded")
    from n0face.detector import FaceDetector
    from n0face.pipeline import Engine

    QSettings("n0face-test", "n0face").clear()
    monkeypatch.setattr(app_window, "QSettings", lambda *a: QSettings("n0face-test", "n0face"))
    w = app_window.MainWindow(Engine(FaceDetector()))
    w.show()
    try:
        for mode in ("obfuscate", "anonymize", "mask", "obfuscate"):
            for b in w.effect_group.buttons():
                b.hover.to(1)
                b.check.to(0.5)
            w._select_mode(mode)
            _process_deletes()
            for _ in range(5):
                clock()._tick()  # must not raise
            assert w.mode_seg.index == list(app_window.MODES).index(mode)
    finally:
        w.close()
        QSettings("n0face-test", "n0face").clear()
