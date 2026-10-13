"""Effect registry."""

import numpy as np

from n0face import plugins
from n0face.effects.base import Effect
from n0face.effects.blocks import GlitchEffect
from n0face.effects.mesh import MeshEffect
from n0face.effects.pixelate import PixelateEffect
from n0face.effects.static import StaticEffect
from n0face.effects.textured import XEffect
from n0face.theme import DEFAULT_THEME, Theme

BUILTIN_EFFECTS = ["mesh", "x", "glitch", "static", "pixelate"]


def effect_names() -> list[str]:
    """Every effect available here: the built-in ones plus masks from add-ons."""
    return [*BUILTIN_EFFECTS[:2], *plugins.registry.masks, *BUILTIN_EFFECTS[2:]]


def make_effects(edges: np.ndarray, theme: Theme = DEFAULT_THEME) -> dict[str, Effect]:
    """`edges` is the face mesh tesselation, as (N, 2) landmark index pairs."""
    effects = {
        "mesh": MeshEffect(edges, theme),
        "x": XEffect(theme),
        "glitch": GlitchEffect(theme),
        "static": StaticEffect(),
        "pixelate": PixelateEffect(),
    }
    for name, factory in plugins.registry.masks.items():
        effects[name] = factory(edges, theme)
    return effects


__all__ = ["BUILTIN_EFFECTS", "Effect", "effect_names", "make_effects"]
