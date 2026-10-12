"""Process photos and recorded videos.

Offline processing can do better than live because it sees the whole file:
- Photos are scanned at full size and in overlapping tiles, to catch small faces.
- Each video frame's mask is the union of detections from the neighboring `window` frames on
  both sides, so a face the detector misses for a few frames stays covered.
- Output carries no metadata (EXIF, GPS, device, timestamps) and no audio unless requested.
- Originals are never modified or overwritten.
"""

import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from n0face.effects import make_effects
from n0face.hud import Hud
from n0face.mask import build_mask
from n0face.modes import MODES
from n0face.pipeline import render_safe
from n0face.theme import DEFAULT_THEME, Theme

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv"}
SUFFIX = ".n0face"
MISS_POLICIES = ("hide", "keep")
TILE_EVERY = 4  # video: run the tiled small-face scan on every Nth frame (must be <= window)
NO_FACE_TEXT = "NO FACE FOUND // HIDDEN"


class Cancelled(Exception):
    pass


@dataclass
class Report:
    input: Path
    output: Path | None = None
    kind: str = ""
    frames: int = 0
    frames_with_faces: int = 0
    hidden_frames: int = 0
    kept_unmasked_frames: int = 0
    max_faces: int = 0
    audio: str = "none"
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"{self.input.name} -> {self.output.name if self.output else '(failed)'}",
            f"{self.frames} frame(s)",
            f"faces in {self.frames_with_faces}",
            f"max {self.max_faces} face(s)",
        ]
        if self.hidden_frames:
            parts.append(f"{self.hidden_frames} hidden (no face)")
        if self.kept_unmasked_frames:
            parts.append(f"{self.kept_unmasked_frames} kept UNMASKED (no face)")
        if self.kind == "video":
            parts.append(f"audio: {self.audio}")
        parts.append(f"{self.seconds:.1f}s")
        return " | ".join(parts)


def media_kind(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def collect_inputs(paths: list[Path]) -> list[Path]:
    """Expand folders (non-recursively) and keep supported files, skipping our own outputs."""
    files = []
    for p in paths:
        candidates = sorted(p.iterdir()) if p.is_dir() else [p]
        files += [c for c in candidates if c.is_file() and media_kind(c) and SUFFIX not in c.stem]
    return files


def output_path(src: Path, out_dir: Path | None = None) -> Path:
    """`photo.jpg` -> `photo.n0face.jpg`; videos always become .mp4. Never overwrites anything."""
    ext = src.suffix.lower() if media_kind(src) == "image" else ".mp4"
    folder = out_dir or src.parent
    candidate = folder / f"{src.stem}{SUFFIX}{ext}"
    n = 1
    while candidate.exists() or candidate.resolve() == src.resolve():
        candidate = folder / f"{src.stem}{SUFFIX}-{n}{ext}"
        n += 1
    return candidate


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def read_image(path: Path) -> np.ndarray:
    """Load as upright BGR. Honors EXIF orientation, drops alpha. Works with non-ASCII paths."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def write_image(path: Path, img: np.ndarray) -> None:
    """Encode with OpenCV, which writes pixel data only: no EXIF, GPS or other metadata."""
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"Could not encode {path.name}")
    buf.tofile(str(path))


def first_frame(path: Path) -> np.ndarray:
    """The image itself, or a video's first frame, as upright BGR."""
    if media_kind(path) == "image":
        return read_image(path)
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {path.name}")
    return frame


class MediaProcessor:
    def __init__(
        self,
        detector,
        mode: str = "anonymize",
        effect: str | None = None,
        theme: Theme = DEFAULT_THEME,
        hud: bool = True,
        on_miss: str = "hide",
        window: int = 4,
        keep_audio: bool = False,
        progress: Callable[[float, np.ndarray | None], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ):
        assert on_miss in MISS_POLICIES
        self.detector = detector
        self.mode = MODES[mode]
        self.effect = make_effects(detector.edges, theme)[effect if effect in self.mode.effects else self.mode.effects[0]]
        self.hud = Hud(theme) if hud else None
        self.on_miss = on_miss
        self.window = window
        self.keep_audio = keep_audio
        self.progress = progress or (lambda *_: None)
        self.cancelled = cancelled or (lambda: False)

    def process(self, src: Path, out_dir: Path | None = None) -> Report:
        kind = media_kind(src)
        if kind is None:
            raise ValueError(f"Unsupported file type: {src.name}")
        dst = output_path(src, out_dir)
        report = Report(input=src, kind=kind)
        start = time.perf_counter()
        if kind == "image":
            self._image(src, dst, report)
        else:
            self._video(src, dst, report)
        report.output = dst
        report.seconds = time.perf_counter() - start
        return report

    # ---- shared ---------------------------------------------------------------------------

    def _mask(self, faces: list[np.ndarray], shape) -> np.ndarray:
        m = self.mode
        return build_mask(faces, shape, margin=m.margin, forehead=m.forehead, silhouette=m.silhouette)

    def _render(self, frame, mask_faces, draw_faces, t, report) -> np.ndarray:
        report.frames += 1
        report.max_faces = max(report.max_faces, len(draw_faces))
        if mask_faces:
            report.frames_with_faces += 1
            return render_safe(frame, self._mask(mask_faces, frame.shape[:2]), draw_faces, False,
                               self.mode, self.effect, t, self.hud)
        if self.on_miss == "keep":
            report.kept_unmasked_frames += 1
            return frame.copy()
        report.hidden_frames += 1
        full = np.full(frame.shape[:2], 255, dtype=np.uint8)
        return render_safe(frame, full, [], True, self.mode, self.effect, t, lost_text=NO_FACE_TEXT)

    def preview(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame exactly as a photo would be, without writing anything."""
        return self.render_frame(frame, self.detector.detect(frame))

    def render_frame(self, frame: np.ndarray, faces: list[np.ndarray]) -> np.ndarray:
        """Render a frame from already-detected faces (used by the cached app preview)."""
        return self._render(frame, faces, faces, 0.0, Report(input=Path()))

    # ---- photos ---------------------------------------------------------------------------

    def _image(self, src: Path, dst: Path, report: Report) -> None:
        img = read_image(src)
        faces = self.detector.detect(img)
        out = self._render(img, faces, faces, 0.0, report)
        write_image(dst, out)
        self.progress(1.0, out)

    # ---- videos ---------------------------------------------------------------------------

    def _video(self, src: Path, dst: Path, report: Report) -> None:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {src.name}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        sink = None
        k = self.window
        pending: deque[tuple[np.ndarray, list]] = deque()
        history: deque[list] = deque(maxlen=k)
        frames_read = 0

        def emit():
            nonlocal sink
            frame, faces = pending.popleft()
            window = [*history, faces, *(f for _, f in pending)]
            union = [lm for fl in window for lm in fl]
            # Draw this frame's own faces; if it had none, borrow the nearest neighbor's.
            draw = faces or next((fl for fl in nearest_neighbors(history, pending) if fl), [])
            out = self._render(frame, union, draw, report.frames / fps, report)
            if sink is None:
                sink = self._open_sink(src, dst, out.shape[1], out.shape[0], fps, report)
            sink.write(out)
            history.append(faces)
            if report.frames % 5 == 0:
                self.progress(min(report.frames / total, 1.0) if total else 0.0, out)

        try:
            while True:
                if self.cancelled():
                    raise Cancelled()
                ok, frame = cap.read()
                if not ok:
                    break
                # Tiled (small-face) detection is the slow part: run it every TILE_EVERY frames.
                # The +-window union carries those detections onto the frames in between.
                tile_now = frames_read % TILE_EVERY == 0
                pending.append((frame, self.detector.detect(frame, tiles=tile_now)))
                frames_read += 1
                if len(pending) > k:
                    emit()
            while pending:
                emit()
            if sink is None:
                raise RuntimeError(f"No frames could be read from {src.name}")
            sink.close()
        except BaseException:
            if sink is not None:
                sink.abort()
            raise
        finally:
            cap.release()
        self.progress(1.0, None)

    def _open_sink(self, src, dst, w, h, fps, report):
        ffmpeg = ffmpeg_path()
        if ffmpeg is None:
            report.warnings.append("ffmpeg not found: wrote MPEG-4 Part 2 video without audio")
            report.audio = "dropped (no ffmpeg)" if self.keep_audio else "none"
            return _OpenCVSink(dst, w, h, fps)
        report.audio = "kept (voice is NOT anonymized)" if self.keep_audio else "removed"
        return _FfmpegSink(ffmpeg, dst, w, h, fps, src if self.keep_audio else None)


class _FfmpegSink:
    """Pipes raw frames into ffmpeg: one H.264 encode, all metadata stripped."""

    def __init__(self, ffmpeg: str, dst: Path, w: int, h: int, fps: float, audio_src: Path | None):
        self.dst = dst
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
        ]
        if audio_src is not None:
            cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "160k", "-shortest"]
        else:
            cmd += ["-map", "0:v:0", "-an"]
        cmd += [
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            # Strip all container/stream metadata (creation time, device, GPS, encoder tags).
            "-map_metadata", "-1", "-map_chapters", "-1",
            "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
            "-movflags", "+faststart",
            str(dst),
        ]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags
        )

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self.close()  # raises with ffmpeg's error message

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace").strip()
        if self.proc.wait() != 0:
            self.dst.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed: {err}")

    def abort(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self.dst.unlink(missing_ok=True)


class _OpenCVSink:
    def __init__(self, dst: Path, w: int, h: int, fps: float):
        self.dst = dst
        self.writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError("Could not open a video writer")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    def abort(self) -> None:
        self.writer.release()
        self.dst.unlink(missing_ok=True)


def nearest_neighbors(history: deque, pending: deque):
    """Neighbor face lists ordered by distance from the current frame: next, previous, ..."""
    past = list(history)[::-1]
    future = [f for _, f in pending]
    for i in range(max(len(past), len(future))):
        if i < len(future):
            yield future[i]
        if i < len(past):
            yield past[i]
