"""Regression: UI strings got double-encoded (UTF-8 saved as Windows-1252), so the window
showed "â€”" instead of "—" and "â€œOBS Virtual Cameraâ€" instead of curly quotes."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
# A UTF-8 lead byte read as Windows-1252 (Â-ô), followed by a continuation byte read the same way.
MOJIBAKE = re.compile("[Â-ô][\u0080-¿ŒœŠšŸŽžƒˆ˜"
                      "–—‘-„†-•…‰‹›€™]")


def test_sources_are_clean_utf8():
    bad = []
    for path in SRC.rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if MOJIBAKE.search(line):
                bad.append(f"{path.relative_to(SRC)}:{n}")
    assert not bad, f"double-encoded text in: {', '.join(bad)}"
