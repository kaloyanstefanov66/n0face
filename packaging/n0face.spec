# PyInstaller spec for the n0face desktop app (one-folder build, no console window).
# Build with:  pyinstaller packaging/n0face.spec --noconfirm   (or run packaging/build.ps1)
# N0FACE_EDITION=pro bundles the n0face PRO add-on (it must be installed); otherwise it's left
# out even if installed, since n0face imports it by name and PyInstaller would follow that.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules

PRO = os.environ.get("N0FACE_EDITION", "free") == "pro"

ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "models" / "face_landmarker.task"), "models"),  # bundled: the app never downloads anything
    (str(ROOT / "models" / "pose_landmarker.task"), "models"),
    (str(ROOT / "assets" / "n0face.ico"), "assets"),
    (str(ROOT / "assets" / "n0face.png"), "assets"),
    (str(ROOT / "assets" / "fonts" / "SedgwickAveDisplay-Regular.ttf"), "assets/fonts"),
    (str(ROOT / "assets" / "fonts" / "SedgwickAveDisplay-OFL.txt"), "assets/fonts"),
    (str(ROOT / "LICENSE"), "."),
]
binaries = collect_dynamic_libs("pyvirtualcam")
hiddenimports = collect_submodules("n0face_pro") if PRO else []
if PRO and not hiddenimports:
    raise SystemExit("N0FACE_EDITION=pro, but n0face_pro isn't installed")

for pkg in ("mediapipe",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyInstaller", *([] if PRO else ["n0face_pro"])],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="n0face",
    icon=str(ROOT / "assets" / "n0face.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="n0face", upx=False)
