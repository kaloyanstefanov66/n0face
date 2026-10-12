"""Color themes for the stylized effects and HUD. Colors are BGR."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    primary: tuple[int, int, int]
    accent: tuple[int, int, int]


THEMES = {
    "cyan": Theme(primary=(255, 230, 40), accent=(200, 40, 255)),
    "magenta": Theme(primary=(200, 40, 255), accent=(255, 230, 40)),
    "amber": Theme(primary=(0, 176, 255), accent=(60, 60, 255)),
    "matrix": Theme(primary=(90, 255, 90), accent=(230, 255, 230)),
    "volt": Theme(primary=(70, 245, 225), accent=(200, 40, 255)),
}
DEFAULT_THEME = THEMES["cyan"]
