"""detect -> mask -> effect. Shared by the desktop app, the CLI and the benchmarks."""

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from n0face import glitch, plugins, wrap
from n0face.effects import Effect, make_effects
from n0face.effects.static import StaticEffect
from n0face.hud import Hud
from n0face.mask import MaskTracker, build_mask
from n0face.modes import MODES, Mode
from n0face.smooth import Steadier
from n0face.theme import DEFAULT_THEME, Theme

FONT = cv2.FONT_HERSHEY_SIMPLEX
# When tracking is lost the whole frame is hidden; never with an effect built from camera pixels.
LOST_FALLBACK = StaticEffect()


def make_tracker(mode: Mode) -> MaskTracker:
    return MaskTracker(
        hold_frames=mode.hold_frames,
        margin=mode.margin,
        forehead=mode.forehead,
        silhouette=mode.silhouette,
        head_fit=mode.geometry,
    )


def anonymize(
    frame: np.ndarray,
    faces: list[np.ndarray],
    tracker: MaskTracker,
    mode: Mode,
    effect: Effect,
    t: float,
    hud: Hud | None = None,
    heads=(),
    draw=None,
) -> tuple[np.ndarray, bool]:
    """The privacy-critical step for live video: return (frame that is safe to publish, lost).

    The mask always comes from the raw `faces` and `heads`. `draw` (a detector Found, e.g.
    steadied over time) replaces what the design is drawn on, when this frame found someone.
    """
    found_now = bool(faces or heads)
    mask, faces, heads, lost = tracker.update(faces, frame.shape[:2], heads)
    if draw is not None and found_now:
        faces, heads = draw.faces, draw.heads
    return render_safe(frame, mask, faces, lost, mode, effect, t, hud, heads=heads), lost


def render_safe(
    frame: np.ndarray,
    mask: np.ndarray,
    faces: list[np.ndarray],
    lost: bool,
    mode: Mode,
    effect: Effect,
    t: float,
    hud: Hud | None = None,
    lost_text: str = "NO FACE IN VIEW // FEED HIDDEN",
    heads=(),
) -> np.ndarray:
    """Apply `effect` inside `mask`. When `lost`, `mask` must cover the whole frame.

    `heads` are the people the pose model found (n0face.heads.Head). Those without a face mesh
    get one fitted from pose in geometry modes, drawn over a cover that even a 3D-wrapped
    design fills completely; other modes cover the whole head.
    """
    if lost and effect.uses_camera_pixels:
        effect = LOST_FALLBACK
    cover = None
    if heads and not lost:
        m = mode
        mask = mask | build_mask([], mask.shape, m.margin, m.forehead, m.silhouette, heads, head_fit=m.geometry)
        if m.geometry:
            drawn = [h.landmarks for h in heads if not h.has_face]
            cover = build_mask(drawn, mask.shape, margin=0.0, forehead=0.0)
            faces = [*faces, *drawn]
    # Only mask mode blurs the face under its design, and never in a fail-closed frame: a
    # blurred whole frame would still show the room.
    blur = mode.geometry and not lost and wrap.config.backing == "blur"
    out = effect.apply(frame, mask, faces if mode.geometry else [], t, cover=cover, blur=blur)
    if mode.geometry and not lost and faces:
        for overlay in plugins.registry.overlays:  # accessories (n0face PRO)
            if overlay.active():
                overlay.draw(out, faces, t, effect.theme or DEFAULT_THEME)
    if glitch.config.enabled and mode.geometry and not lost:
        x, y, w, h = cv2.boundingRect(mask)
        if w:
            glitch.apply(out, (x, y, x + w, y + h), t, effect.theme or DEFAULT_THEME)
    if lost:
        banner(out, lost_text, (255, 230, 40))
    elif hud is not None and mode.hud:
        hud.draw(out, faces, t)
    return out


def banner(img: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    scale = max(img.shape[1] / 1280, 0.4)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = (img.shape[1] - tw) // 2, int(40 * scale) + th
    cv2.rectangle(img, (x - 14, y - th - 12), (x + tw + 14, y + 12), (0, 0, 0), -1)
    cv2.rectangle(img, (x - 14, y - th - 12), (x + tw + 14, y + 12), color, 1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


@dataclass
class Stats:
    fps: float = 0.0
    detect_ms: float = 0.0
    render_ms: float = 0.0
    faces: int = 0
    lost: bool = True
    fail_closed_frames: int = 0


class Engine:
    """Holds the processing state. Settings may be changed from another thread (the UI)."""

    def __init__(self, detector, mode: str = "mask", effect: str | None = None, theme: Theme = DEFAULT_THEME, hud: bool = True):
        self.detector = detector
        self._lock = threading.Lock()
        self.theme = theme
        self.effects = make_effects(detector.edges, theme)
        self.hud = Hud(theme)
        self.hud_enabled = hud
        self.stats = Stats()
        self.steadier = Steadier()
        self._start = time.perf_counter()
        self._last = self._start
        self.set_mode(mode, effect)

    @property
    def mode(self) -> Mode:
        return MODES[self.mode_name]

    def set_mode(self, name: str, effect: str | None = None) -> None:
        with self._lock:
            self.mode_name = name
            self.effect_name = effect if effect in MODES[name].effects else MODES[name].effects[0]
            self.tracker = make_tracker(MODES[name])

    def set_effect(self, name: str) -> None:
        with self._lock:
            if name in self.mode.effects:
                self.effect_name = name

    def set_theme(self, theme: Theme) -> None:
        with self._lock:
            self.theme = theme
            self.effects = make_effects(self.detector.edges, theme)
            self.hud = Hud(theme, session=self.hud.session)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return the publishable version of `frame` and update `stats`."""
        with self._lock:
            now = time.perf_counter()
            t = now - self._start
            found = self.detector.detect(frame, int(t * 1000))
            faces = found.faces
            draw = self.steadier(found, t) if self.mode.geometry else None
            t1 = time.perf_counter()
            safe, lost = anonymize(
                frame, faces, self.tracker, self.mode, self.effects[self.effect_name], t,
                self.hud if self.hud_enabled else None, found.heads, draw,
            )
            t2 = time.perf_counter()

            s = self.stats
            s.fps = 0.9 * s.fps + 0.1 / max(now - self._last, 1e-6)
            s.detect_ms, s.render_ms = (t1 - now) * 1000, (t2 - t1) * 1000
            s.faces, s.lost = len(faces), lost
            s.fail_closed_frames += lost
            self._last = now
            return safe
