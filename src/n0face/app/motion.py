"""Spring physics for UI motion.

One shared clock drives every active spring and stops when nothing is moving, so an idle UI
costs nothing. Springs are specified like SwiftUI's: `response` (seconds, roughly the period)
and `damping` (fraction of critical; 1 = no overshoot).

With reduced motion on, springs jump straight to their target.
"""

import math
import sys
import time
import traceback
from collections.abc import Callable

import shiboken6
from PySide6.QtCore import QObject, QTimer

# Presets (response s, damping fraction).
SNAPPY = (0.28, 0.82)  # tabs, segmented pills, selection
BOUNCY = (0.38, 0.62)  # toggle knobs, presses
SMOOTH = (0.55, 0.92)  # fades, reveals, camera
GENTLE = (0.9, 1.0)  # ambient blends


class _Clock(QObject):
    def __init__(self):
        super().__init__()
        self.springs: set["Spring"] = set()
        self.timer = QTimer(self, interval=16, timeout=self._tick)
        self.last = time.perf_counter()

    def add(self, spring: "Spring") -> None:
        if not self.springs:
            self.last = time.perf_counter()
            self.timer.start()
        self.springs.add(spring)

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(now - self.last, 1 / 20)  # never take a giant step after a stall
        self.last = now
        for s in list(self.springs):
            # A spring can outlive its widget (e.g. a row deleted mid-hover-fade). Drop it
            # rather than touch a deleted object; one bad spring must never stall the rest.
            if not s.alive():
                self.springs.discard(s)
                continue
            try:
                if s._step(dt):
                    self.springs.discard(s)
                s._notify()
            except Exception:  # noqa: BLE001 - keep the UI animating; report and drop this spring
                self.springs.discard(s)
                traceback.print_exc()
        if not self.springs:
            self.timer.stop()


_clock: _Clock | None = None
_reduced = False


def clock() -> _Clock:
    global _clock
    if _clock is None:
        _clock = _Clock()
    return _clock


def set_reduced_motion(on: bool) -> None:
    global _reduced
    _reduced = on


def reduced_motion() -> bool:
    return _reduced


def system_prefers_reduced_motion() -> bool:
    """Windows 'Show animations in Windows' off -> reduce motion."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        SPI_GETCLIENTAREAANIMATION = 0x1042
        value = ctypes.c_int(1)
        ctypes.windll.user32.SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(value), 0)
        return value.value == 0
    except Exception:  # noqa: BLE001 - a preference lookup must never break startup
        return False


class Spring:
    """Animates a float toward `target`. `on_change` runs after every step (e.g. widget.update)."""

    def __init__(self, value: float = 0.0, preset=SNAPPY, on_change: Callable[[], None] | None = None):
        self.value = self.target = float(value)
        self.velocity = 0.0
        self.on_change = on_change
        self.set_preset(preset)

    def set_preset(self, preset) -> None:
        response, damping = preset
        omega = 2 * math.pi / response
        self.k = omega * omega
        self.c = 2 * damping * omega

    def to(self, target: float, velocity: float | None = None) -> None:
        self.target = float(target)
        if velocity is not None:
            self.velocity = velocity
        if _reduced:
            self.snap(target)
        elif not self.settled() and self.alive():
            clock().add(self)

    def snap(self, value: float) -> None:
        self.value = self.target = float(value)
        self.velocity = 0.0
        clock().springs.discard(self)
        if self.alive():
            self._notify()

    def alive(self) -> bool:
        """False once the Qt object behind `on_change` (usually a widget's update) is deleted."""
        owner = getattr(self.on_change, "__self__", None)
        if owner is None:
            return True
        try:
            return shiboken6.isValid(owner)
        except TypeError:  # not a Qt object
            return True

    def settled(self) -> bool:
        return abs(self.value - self.target) < 1e-3 and abs(self.velocity) < 1e-3

    @property
    def moving(self) -> bool:
        return self in clock().springs

    def _step(self, dt: float) -> bool:
        # Semi-implicit Euler in small sub-steps: stable for stiff springs.
        steps = max(1, int(dt / 0.004))
        h = dt / steps
        for _ in range(steps):
            a = -self.k * (self.value - self.target) - self.c * self.velocity
            self.velocity += a * h
            self.value += self.velocity * h
        if self.settled():
            self.value, self.velocity = self.target, 0.0
            return True
        return False

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()


def mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_cubic(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3
