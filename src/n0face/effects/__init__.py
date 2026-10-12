"""Effect registry."""

import numpy as np

from n0face.effects.base import Effect
from n0face.effects.blocks import GlitchEffect
from n0face.effects.led import LedEffect
from n0face.effects.mesh import MeshEffect
from n0face.effects.pixelate import PixelateEffect
from n0face.effects.static import StaticEffect
from n0face.effects.textured import CircuitEffect, HexEffect
from n0face.theme import DEFAULT_THEME, Theme

EFFECT_NAMES = ["mesh", "led", "circuit", "hex", "glitch", "static", "pixelate"]


def make_effects(edges: np.ndarray, theme: Theme = DEFAULT_THEME) -> dict[str, Effect]:
    """`edges` is the face mesh tesselation, as (N, 2) landmark index pairs."""
    return {
        "mesh": MeshEffect(edges, theme),
        "led": LedEffect(theme),
        "circuit": CircuitEffect(theme),
        "hex": HexEffect(theme),
        "glitch": GlitchEffect(theme),
        "static": StaticEffect(),
        "pixelate": PixelateEffect(),
    }


__all__ = ["EFFECT_NAMES", "Effect", "make_effects"]
