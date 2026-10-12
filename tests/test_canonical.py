import numpy as np
import pytest

from n0face.detector import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not downloaded")


def test_canonical_mesh_loads_from_model_bundle():
    from n0face import canonical

    face = canonical.load()
    assert face.vertices.shape == (468, 3)
    assert face.triangles.shape == (898, 3)
    assert face.triangles.max() == 467
    # Nose tip (landmark 1) is the most forward point of a face.
    assert face.vertices[:, 2].argmax() in (1, 4, 5, 195)


def test_preview_renders_every_effect_and_inspects_landmarks():
    from n0face.app.preview3d import MaskScene, Pose, focal, project_points
    from n0face.modes import MODES
    from n0face.theme import DEFAULT_THEME

    scene = MaskScene()
    w, h = 160, 140
    for mode in MODES.values():
        for effect in mode.effects:
            frame = scene.render(w, h, effect, mode.name, DEFAULT_THEME, 1.0, Pose(0.3, 0.1))
            img = frame.image
            assert img.ndim == 3 and img.shape[2] == 4 and img[..., 3].any()
            assert (img[..., :3] <= img[..., 3:4]).all()  # valid premultiplied alpha
            # The ROI stays inside the canvas.
            assert 0 <= frame.x and frame.x + img.shape[1] <= w
            assert 0 <= frame.y and frame.y + img.shape[0] <= h

    # Hovering exactly over the projected nose tip reports landmark 1.
    pose = Pose(0.2, 0.0)
    from n0face.app.preview3d import rotation

    _, p2 = project_points(scene.v[[1]], rotation(pose.yaw, pose.pitch), w, h, focal(w, h, pose.zoom))
    frame = scene.render(w, h, "mesh", "mask", DEFAULT_THEME, 1.0, pose, hover=tuple(p2[0]))
    assert frame.info["index"] == 1
    frame = scene.render(w, h, "glitch", "anonymize", DEFAULT_THEME, 1.0, pose, hover=tuple(p2[0]))
    assert frame.info == {"hidden": True}
