import numpy as np

from n0face.detector import Found
from n0face.smooth import Steadier, steady


def _shaky(face, n, speed=0.0, noise=6.0, seed=0):
    """`n` frames of `face` moving right at `speed` px/frame, each shaken by `noise` px."""
    rng = np.random.default_rng(seed)
    return [face + (speed * i, 0) + rng.normal(0, noise, 2).astype(np.float32) for i in range(n)]


def _center(found):
    return found.faces[0][:468].mean(axis=0)


def _steady_errors(face, noise):
    frames = _shaky(face, 31, speed=5.0, noise=noise)
    truth = [face.mean(axis=0) + (5.0 * i, 0) for i in range(31)]
    raw, out = [], []
    for i in range(6, 25):
        nb = [(j, Found([frames[i + j]])) for j in range(-6, 7) if j]
        raw.append(np.linalg.norm(frames[i].mean(axis=0) - truth[i]))
        out.append(np.linalg.norm(_center(steady(Found([frames[i]]), nb, 2.4)) - truth[i]))
    return np.mean(raw), np.mean(out)


def test_steady_removes_jitter(face):
    raw, out = _steady_errors(face, noise=6.0)
    assert out < 0.6 * raw


def test_steady_does_not_lag_behind_motion(face):
    _, out = _steady_errors(face, noise=0.0)
    assert out < 0.5  # looking both ways: a steadily moving face stays exactly under its mask


def test_steady_never_mixes_two_people(face):
    other = face + (400, 0)
    out = steady(Found([face]), [(1, Found([other])), (-1, Found([other]))], 2.0)
    assert np.allclose(out.faces[0], face)


def test_live_steadier_calms_a_still_head_and_follows_a_fast_one(face):
    s = Steadier()
    still = _shaky(face, 60, noise=3.0)
    out = [_center(s(Found([f]), i / 30)) for i, f in enumerate(still)]
    raw = [f.mean(axis=0) for f in still]
    assert np.std(np.array(out[20:]), axis=0).max() < 0.5 * np.std(np.array(raw[20:]), axis=0).max()

    s = Steadier()
    moving = _shaky(face, 30, speed=20.0, noise=1.0)  # 20 px/frame at 30 fps: a quick turn
    out = [_center(s(Found([f]), i / 30)) for i, f in enumerate(moving)]
    lag = np.linalg.norm(moving[-1].mean(axis=0) - out[-1])
    assert lag < 40  # under two frames behind
