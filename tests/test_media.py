import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from conftest import H, W, FakeDetector
from n0face.detector import tile_grid
from n0face.media import Cancelled, MediaProcessor, collect_inputs, output_path, read_image

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _photo() -> np.ndarray:
    img = np.full((H, W, 3), (60, 80, 100), np.uint8)
    cv2.circle(img, (320, 240), 110, (120, 150, 190), -1)
    return img


def _jpeg_with_exif(path, img):
    """Write a JPEG carrying an EXIF block that contains a recognizable secret."""
    _, buf = cv2.imencode(".jpg", img)
    data = buf.tobytes()
    payload = b"Exif\x00\x00" + b"II*\x00\x08\x00\x00\x00" + b"\x00\x00" + b"\x00\x00\x00\x00" + b"GPS-SECRET-51.5N"
    app1 = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    path.write_bytes(data[:2] + app1 + data[2:])


# ---- paths --------------------------------------------------------------------------------


def test_output_path_never_overwrites(tmp_path):
    src = tmp_path / "pic.JPG"
    src.write_bytes(b"x")
    first = output_path(src)
    assert first.name == "pic.n0face.jpg"
    first.write_bytes(b"y")
    assert output_path(src).name == "pic.n0face-1.jpg"
    assert output_path(tmp_path / "clip.mov").suffix == ".mp4"


def test_collect_inputs_filters_and_skips_outputs(tmp_path):
    for name in ("a.jpg", "b.mp4", "notes.txt", "a.n0face.jpg"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in collect_inputs([tmp_path])] == ["a.jpg", "b.mp4"]


def test_tile_grid_covers_large_images():
    w, h = 4000, 3000
    tiles = tile_grid(w, h)
    cover = np.zeros((h, w), bool)
    for x, y, tw, th in tiles:
        cover[y : y + th, x : x + tw] = True
    assert cover.all() and len(tiles) > 1
    assert tile_grid(400, 300) == []


# ---- photos -------------------------------------------------------------------------------


def test_photo_masked_metadata_stripped_original_untouched(tmp_path, face):
    src = tmp_path / "photo.jpg"
    _jpeg_with_exif(src, _photo())
    original = src.read_bytes()
    assert b"GPS-SECRET" in original

    report = MediaProcessor(FakeDetector([face]), mode="anonymize").process(src)

    assert src.read_bytes() == original
    data = report.output.read_bytes()
    assert b"Exif" not in data and b"GPS-SECRET" not in data
    out, before = read_image(report.output).astype(int), read_image(src).astype(int)
    cx, cy = np.round(face.mean(axis=0)).astype(int)
    assert np.abs(out[cy - 5 : cy + 5, cx - 5 : cx + 5] - before[cy - 5 : cy + 5, cx - 5 : cx + 5]).mean() > 20
    assert np.abs(out[:, :30] - before[:, :30]).mean() < 3  # background kept (JPEG noise only)
    assert report.frames == report.frames_with_faces == 1


@pytest.mark.parametrize("policy", ["hide", "keep"])
def test_photo_without_face_follows_miss_policy(tmp_path, policy):
    src = tmp_path / "empty.png"
    cv2.imwrite(str(src), _photo())
    report = MediaProcessor(FakeDetector([]), on_miss=policy).process(src)
    out = cv2.imread(str(report.output))
    if policy == "hide":
        assert report.hidden_frames == 1
        assert not any(np.array_equal(out[y], _photo()[y]) for y in range(H))
    else:
        assert report.kept_unmasked_frames == 1
        assert np.array_equal(out, _photo())


# ---- videos -------------------------------------------------------------------------------


def _make_video(path, frames=20, audio=False, title=None):
    """Lossless test clip via ffmpeg, optionally with an audio track and metadata tags."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "10", "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", "-shortest"]
    if title:
        cmd += ["-metadata", f"title={title}", "-metadata", "location=+51.5074-000.1278/"]
    cmd += ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)]
    raw = b"".join(_photo().tobytes() for _ in range(frames))
    subprocess.run(cmd, input=raw, check=True)


def _probe(path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, check=True,
    ).stdout
    return json.loads(out)


def _frame_count(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


@needs_ffmpeg
def test_video_window_covers_short_detector_misses(tmp_path, face):
    src = tmp_path / "clip.mp4"
    _make_video(src, frames=20)
    misses = {8, 9, 10}
    detector = FakeDetector(lambda i: [] if i in misses else [face])

    report = MediaProcessor(detector, mode="obfuscate", window=4).process(src)

    assert report.frames == 20 and report.frames_with_faces == 20 and report.hidden_frames == 0
    assert _frame_count(report.output) == 20


@needs_ffmpeg
def test_video_frames_with_only_a_head_are_masked_not_hidden(tmp_path, head):
    """Looking down: the face model sees nothing but the pose model finds the head."""
    src = tmp_path / "clip.mp4"
    _make_video(src, frames=8)
    report = MediaProcessor(FakeDetector([], heads=[head]), mode="mask").process(src)
    assert report.hidden_frames == 0 and report.frames_with_faces == 8


@needs_ffmpeg
def test_video_long_miss_is_hidden(tmp_path, face):
    src = tmp_path / "clip.mp4"
    _make_video(src, frames=20)
    detector = FakeDetector(lambda i: [face] if i < 5 else [])
    report = MediaProcessor(detector, window=2).process(src)
    # Frames 0-6 have a detection within +-2 frames; 7..19 don't.
    assert report.frames_with_faces == 7 and report.hidden_frames == 13


@needs_ffmpeg
def test_video_strips_metadata_and_audio_by_default(tmp_path, face):
    src = tmp_path / "clip.mp4"
    _make_video(src, audio=True, title="SECRET-TITLE")
    assert any(s["codec_type"] == "audio" for s in _probe(src)["streams"])

    report = MediaProcessor(FakeDetector([face])).process(src)

    info = _probe(report.output)
    assert [s["codec_type"] for s in info["streams"]] == ["video"]
    tags = json.dumps(info["format"].get("tags", {})) + json.dumps([s.get("tags", {}) for s in info["streams"]])
    assert "SECRET-TITLE" not in tags and "51.5074" not in tags
    assert report.audio == "removed"


@needs_ffmpeg
def test_video_keep_audio(tmp_path, face):
    src = tmp_path / "clip.mp4"
    _make_video(src, audio=True)
    report = MediaProcessor(FakeDetector([face]), keep_audio=True).process(src)
    assert sorted(s["codec_type"] for s in _probe(report.output)["streams"]) == ["audio", "video"]
    assert "NOT anonymized" in report.audio


@needs_ffmpeg
def test_cancel_leaves_no_output(tmp_path, face):
    src = tmp_path / "clip.mp4"
    _make_video(src)
    detector = FakeDetector([face])
    processor = MediaProcessor(detector, cancelled=lambda: detector.calls >= 5)
    with pytest.raises(Cancelled):
        processor.process(src)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]
