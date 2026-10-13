"""Add-ons: the hook n0face PRO (or any plugin) uses to add masks, overlays and UI.

n0face itself never needs a plugin. At startup `load()` imports the `n0face_pro` package if
it's installed and lets it register what it brings:

- masks: an Effect factory (and its design in `n0face.designs.DESIGNS`), added to mask mode
- overlays: things drawn on heads after the mask (accessories), see `Overlay`
- pages: sidebar pages, built by the app window (`pages["gear"]`)
- features: switches the app checks with `has()`, e.g. "glitch_knobs"

Plugins that interact with n0face only through this interface may use any license; see
LICENSE-PLUGIN-EXCEPTION.md. Nothing here touches the network.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

# Where the EXTRAS tab's support button points (opened in the browser; n0face itself stays offline).
PRO_URL = "https://github.com/kaloyanstefanov66/n0face#n0face-pro"


class Overlay(Protocol):
    """Something drawn on each head in mask mode, after the mask and before GLITCH."""

    def active(self) -> bool: ...

    def draw(self, canvas, faces, t: float, theme) -> None:
        """On a frame: `faces` are (478, 2) face meshes in pixels."""

    def points_cm(self, t: float) -> list:
        """3D points (canonical head, cm) it may reach, so the 3D preview can make room."""

    def draw_projected(self, canvas, project, px_per_cm: float, t: float, theme) -> None:
        """On the 3D preview: `project(points_cm) -> (pixels, depth)`."""


@dataclass
class Registry:
    masks: dict[str, Callable] = field(default_factory=dict)  # name -> factory(edges, theme) -> Effect
    overlays: list[Overlay] = field(default_factory=list)
    pages: dict[str, Callable] = field(default_factory=dict)  # name -> build(window) -> QWidget
    features: set[str] = field(default_factory=set)
    name: str = ""  # e.g. "n0face PRO 1.0"


registry = Registry()
_loaded = False


def load() -> Registry:
    """Load installed add-ons once. Safe to call more than once."""
    global _loaded
    if _loaded:
        return registry
    _loaded = True
    try:
        import n0face_pro
    except ImportError:
        return registry
    n0face_pro.register(registry)
    if registry.masks:
        from n0face.modes import MODES

        mask = MODES["mask"]
        extra = tuple(n for n in registry.masks if n not in mask.effects)
        MODES["mask"] = replace(mask, effects=mask.effects + extra)
    return registry


def pro() -> bool:
    return bool(registry.name)


def has(feature: str) -> bool:
    return feature in registry.features
