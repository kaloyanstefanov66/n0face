"""Locate bundled files (face model, icon) both from a source checkout and inside the packaged app."""

import sys
from pathlib import Path


def base_dir() -> Path:
    if getattr(sys, "frozen", False):  # PyInstaller bundle: files sit next to the bundled modules
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]  # repo root (src/n0face/resources.py)


def resource(*parts: str) -> Path:
    return base_dir().joinpath(*parts)
