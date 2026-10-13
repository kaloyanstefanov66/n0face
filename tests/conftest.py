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


@pytest.fixture
def head():
    """Pose head keypoints (n0face.heads.Head) of a frontal canonical face at (320, 250),
    about 190 px wide. Needs the face model file (the canonical mesh lives in it)."""
    from n0face.detector import DEFAULT_MODEL
    from n0face.heads import Head, _canonical

    if not DEFAULT_MODEL.exists():
        pytest.skip("model not downloaded")
    _, ref = _canonical()
    return Head((np.array([320.0, 250.0]) + ref[:, :2] * 12).astype(np.float32))


class FakeDetector:
    """Stands in for the MediaPipe detectors. `faces` is a list, or a callable(call_index);
    `heads` likewise (people found by pose without a face mesh)."""

    edges = np.random.default_rng(1).integers(0, 478, size=(800, 2))

    def __init__(self, faces, heads=()):
        self.faces = faces
        self.heads = heads
        self.calls = 0

    def detect(self, frame, timestamp_ms=0, **_):
        from n0face.detector import Found

        faces = self.faces(self.calls) if callable(self.faces) else self.faces
        heads = self.heads(self.calls) if callable(self.heads) else self.heads
        self.calls += 1
        return Found(list(faces), list(heads))
