"""Privacy modes. Each one states plainly what it does and does not protect against.

Streaming to the virtual camera is available in every mode; it is an output, not a mode.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    name: str
    effects: tuple[str, ...]
    # Whether effects may receive landmark positions. Rendering landmarks reproduces your
    # facial geometry (eye spacing, jaw shape, expressions), which is itself identifying.
    geometry: bool
    margin: float
    forehead: float
    silhouette: str  # "hull" follows the face outline, "ellipse" hides it
    hold_frames: int
    hud: bool
    claim: str


MODES = {
    "mask": Mode(
        name="mask",
        effects=("mesh", "led", "circuit", "hex"),
        geometry=True,
        margin=0.15,
        forehead=0.3,
        silhouette="hull",
        hold_frames=6,
        hud=True,
        claim="DECORATIVE. Replaces face pixels but renders your facial geometry and expressions. "
        "Not identity protection.",
    ),
    "obfuscate": Mode(
        name="obfuscate",
        effects=("glitch", "static", "pixelate"),
        geometry=False,
        margin=0.2,
        forehead=0.35,
        silhouette="hull",
        hold_frames=6,
        hud=False,
        claim="Hides face pixels (pixelate is weakest: it is derived from them). Face outline, hair, "
        "body, voice, clothing and background can still identify you.",
    ),
    "anonymize": Mode(
        name="anonymize",
        effects=("glitch", "static"),
        geometry=False,
        margin=0.45,
        forehead=0.7,
        silhouette="ellipse",
        hold_frames=15,
        hud=False,
        claim="Strongest available: covers the whole head with a shape-hiding ellipse, renders no "
        "geometry, holds the mask longer on dropouts. Body, voice and surroundings are NOT hidden.",
    ),
}
MODE_NAMES = list(MODES)
