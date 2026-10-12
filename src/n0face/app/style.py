"""Design tokens and the Qt stylesheet.

Near-black, slightly teal surfaces; hairline borders; monospace uppercase labels; one accent
color from the active theme. Panels are translucent so the ambient background breathes through.
"""

from PySide6.QtGui import QColor, QFont

from n0face.theme import Theme

BG = "#05080a"
PANEL = "rgba(9, 15, 19, 0.9)"  # near-opaque: the ambient layer can animate without repainting it
PANEL_SOLID = "#0a1014"
LINE = "#14212a"
LINE_STRONG = "#203540"
TEXT = "#cfe2e8"
DIM = "#5e7781"
FAINT = "#33464f"
LIVE = "#ff3b4e"
OK = "#3dff9a"
WARN = "#ffb347"

MONO_FAMILIES = ["JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo", "monospace"]


def mono(size: float = 9, bold: bool = False, spacing: float = 1.4) -> QFont:
    f = QFont()
    f.setFamilies(MONO_FAMILIES)
    f.setPointSizeF(size)
    f.setBold(bold)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def hex_rgb(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def accent_color(theme: Theme) -> QColor:
    return QColor(hex_rgb(theme.primary))


def with_alpha(color: QColor | str, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def stylesheet(theme: Theme) -> str:
    a = hex_rgb(theme.primary)
    ac = accent_color(theme)
    a_soft = f"rgba({ac.red()}, {ac.green()}, {ac.blue()}, 0.10)"
    a_mid = f"rgba({ac.red()}, {ac.green()}, {ac.blue()}, 0.45)"
    return f"""
    * {{ color: {TEXT}; outline: none; }}
    QToolTip {{ background: {PANEL_SOLID}; color: {TEXT}; border: 1px solid {LINE_STRONG}; padding: 4px 6px; }}

    #Sidebar {{ background: {PANEL}; border: 1px solid {LINE}; }}
    #Page {{ background: transparent; }}
    QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 6px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {LINE_STRONG}; min-height: 24px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

    #Dim {{ color: {DIM}; }}
    #Faint {{ color: {FAINT}; }}
    #Body {{ color: {DIM}; }}

    QPushButton {{
        background: transparent; border: 1px solid {LINE_STRONG}; padding: 9px 12px; text-align: left;
    }}
    QPushButton:hover {{ border-color: {a_mid}; }}
    QPushButton:checked {{ border-color: {a}; color: {a}; background: {a_soft}; }}
    QPushButton:disabled {{ color: {FAINT}; border-color: {LINE}; }}
    #Segment {{ text-align: center; padding: 9px 6px; }}
    #EffectRow {{ padding: 11px 14px; }}
    #IconBtn {{ padding: 7px; text-align: center; }}
    #WinBtn {{ border: none; padding: 6px 12px; color: {DIM}; text-align: center; background: transparent; }}
    #WinBtn:hover {{ color: {TEXT}; }}
    #CloseBtn {{ border: none; padding: 6px 12px; color: {DIM}; text-align: center; background: transparent; }}
    #CloseBtn:hover {{ color: {LIVE}; }}

    QLineEdit {{ background: rgba(0,0,0,0.25); border: 1px solid {LINE_STRONG}; padding: 8px 10px; color: {TEXT}; }}
    QComboBox {{ background: rgba(0,0,0,0.25); border: 1px solid {LINE_STRONG}; padding: 8px 10px; }}
    QComboBox:hover {{ border-color: {a_mid}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{ background: {PANEL_SOLID}; border: 1px solid {LINE_STRONG}; selection-background-color: {LINE_STRONG}; }}
    """
