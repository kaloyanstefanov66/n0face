"""Smoke test of the real GUI entry point, so a broken import can't ship silently again."""

import os

import pytest

from n0face.detector import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not downloaded")


def test_gui_entry_point_builds_window(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import n0face.netguard as netguard

    unlocks = []
    real_lock = netguard.lock_network
    monkeypatch.setattr(netguard, "lock_network", lambda: unlocks.append(real_lock()))

    app = QApplication.instance() or QApplication([])
    from PySide6.QtCore import QSettings

    from n0face.app import main as app_main
    from n0face.app import window as app_window

    # Never touch the user's real preferences from tests.
    QSettings("n0face-test", "n0face").clear()
    monkeypatch.setattr(app_window, "QSettings", lambda *a: QSettings("n0face-test", "n0face"))

    app.setWindowIcon(app_main.make_icon())
    window, detector = app_main.build_window(load_plugins=False)
    try:
        window.show()
        app.processEvents()
        assert window.isVisible()
        assert unlocks, "the app must lock the network at startup"
    finally:
        window.close()
        detector.close()
        for unlock in unlocks:
            unlock()
