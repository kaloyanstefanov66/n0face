"""One-time download of the MediaPipe face landmarker model into models/.

This is the only network access n0face ever does; at runtime it reads the local file.
"""

from pathlib import Path
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEST = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def main() -> None:
    if DEST.exists():
        print(f"Model already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
