"""Mask designs that only exist as designs: flat stickers, or wrapped around the face in 3D."""

from n0face.effects.base import PANEL, Effect
from n0face.theme import DEFAULT_THEME, Theme


class TexturedEffect(Effect):
    always_textured = True

    def __init__(self, theme: Theme = DEFAULT_THEME):
        self.theme = theme

    def render(self, h, w, faces, t, rng):
        # No face to map onto (e.g. fail-closed full-frame cover): plain dark panel.
        return PANEL.render(h, w, faces, t, rng)


class XEffect(TexturedEffect):
    """Graffiti purge mask: painted X eyes and a stitched-shut grin, with paint drips."""

    name = "x"
    wrap_design = "x"
