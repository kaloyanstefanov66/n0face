# PyInstaller spec for the n0face desktop app (one-folder build, no console window).
# Build with:  pyinstaller packaging/n0face.spec --noconfirm   (or run packaging/build.ps1)

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "models" / "face_landmarker.task"), "models"),  # bundled: the app never downloads anything
    (str(ROOT / "assets" / "n0face.ico"), "assets"),
    (str(ROOT / "assets" / "n0face.png"), "assets"),
    (str(ROOT / "LICENSE"), "."),
]
binaries = collect_dynamic_libs("pyvirtualcam")
hiddenimports = ["glcontext", "glcontext.wgl"]

for pkg in ("mediapipe", "moderngl"):
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
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyInstaller"],
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
