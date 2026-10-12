import numpy as np
import pytest

H, W = 480, 640


@pytest.fixture
def face() -> np.ndarray:
    """A plausible fake 478-point face centered at (320, 240)."""
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, 478)
    rad = np.sqrt(rng.uniform(0, 1, 478))
    lm = np.stack([320 + 80 * rad * np.cos(ang), 240 + 100 * rad * np.sin(ang)], axis=1)
    named = {
        33: (270, 220), 133: (305, 220), 159: (287, 212), 145: (287, 228),
        362: (335, 220), 263: (370, 220), 386: (352, 212), 374: (352, 228),
        61: (295, 290), 291: (345, 290), 13: (320, 288), 14: (320, 296),
    }
    for i, p in named.items():
        lm[i] = p
    return lm.astype(np.float32)


class FakeDetector:
    """Stands in for the MediaPipe detectors. `faces` is a list, or a callable(call_index)."""

    edges = np.random.default_rng(1).integers(0, 478, size=(800, 2))

    def __init__(self, faces):
        self.faces = faces
        self.calls = 0

    def detect(self, frame, timestamp_ms=0, **_):
        result = self.faces(self.calls) if callable(self.faces) else self.faces
        self.calls += 1
        return result
