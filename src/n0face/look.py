"""Cosmetic overrides for the mask designs: per-element colours, swappable parts, stroke and
glow. Defaults leave every design as drawn; n0face PRO's CUSTOMIZE panel sets them.

Elements are shared by all designs, each mapping them onto its own parts:
- eyes, mouth: the features (X eyes and stitches, LED eyes, visor, sockets...)
- lines: the main linework (mesh, traces, plate edges, crystal edges)
- accent: small highlights (mesh nodes, pulses, sparks, vertices)
- base: the surface (LED dots, hex plates, skull bone, facets)
"""

from dataclasses import dataclass, field

ELEMENTS = ("eyes", "mouth", "lines", "accent", "base")


@dataclass
class Look:
    colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # element -> BGR
    parts: dict[str, str] = field(default_factory=dict)  # slot ("eyes", "mouth") -> variant
    stroke: float = 1.0  # line thickness
    glow: float = 1.0  # glow strength
    drips: bool = True  # paint drips on the X mask


look = Look()


def color(element: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    return look.colors.get(element, default)


def part(slot: str, default: str) -> str:
    return look.parts.get(slot, default)
