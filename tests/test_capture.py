import numpy as np
import pytest

import n0face.capture as capture


class FakeCap:
    """Opens fine; delivers frames only if `works`."""

    opened: list[str] = []

    def __init__(self, name, works):
        self.name, self.works = name, works
        FakeCap.opened.append(name)

    def isOpened(self):  # noqa: N802
        return True

    def set(self, *_):
        return True

    def get(self, prop):
        return {3: 1280, 4: 720}.get(prop, 0)

    def read(self):
        return (True, np.zeros((720, 1280, 3), np.uint8)) if self.works else (False, None)

    def release(self):
        pass


@pytest.fixture
def fake_backends(monkeypatch):
    FakeCap.opened = []
    monkeypatch.setattr(capture, "WARMUP_S", 0.1)
    monkeypatch.setattr(capture, "_backends", lambda: (("MSMF", 1), ("DSHOW", 2)))

    def install(working: set[str]):
        names = {1: "MSMF", 2: "DSHOW"}
        monkeypatch.setattr(capture.cv2, "VideoCapture", lambda src, api: FakeCap(names[api], names[api] in working))

    return install


def test_prefers_first_backend_that_delivers_frames(fake_backends):
    fake_backends({"MSMF", "DSHOW"})
    cam = capture.Camera(0)
    assert cam.backend == "MSMF" and cam.read().shape == (720, 1280, 3)


def test_falls_back_when_a_backend_opens_but_gives_no_frames(fake_backends):
    fake_backends({"DSHOW"})  # MSMF "opens" but never produces a frame (camera busy)
    cam = capture.Camera(0)
    assert cam.backend == "DSHOW"
    assert FakeCap.opened == ["MSMF", "DSHOW"]


def test_clear_error_when_no_backend_delivers(fake_backends):
    fake_backends(set())
    with pytest.raises(RuntimeError, match="another app"):
        capture.Camera(0)
