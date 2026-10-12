# Build the n0face Windows installer: icon -> PyInstaller bundle -> Inno Setup installer.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File packaging\build.ps1
# Output: dist\n0face\n0face.exe (portable folder) and dist\n0face-setup-<version>.exe

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path "models\face_landmarker.task")) { & $python scripts\get_model.py }
& $python packaging\make_icon.py

& $python -m PyInstaller packaging\n0face.spec --noconfirm --clean --distpath dist --workpath build
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$version = (Select-String -Path pyproject.toml -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found (winget install JRSoftware.InnoSetup)" }
& $iscc "/DAppVersion=$version" packaging\n0face.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

Get-Item "dist\n0face-setup-$version.exe" | Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
