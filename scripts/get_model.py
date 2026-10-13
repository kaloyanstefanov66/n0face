"""One-time download of the MediaPipe models into models/: face landmarks and pose.

This is the only network access n0face ever does; at runtime it reads the local files.
"""

from pathlib import Path
import urllib.request

BASE = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {
    "face_landmarker.task": BASE + "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker.task": BASE + "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dest = DIR / name
        if dest.exists():
            print(f"Model already present: {dest}")
            continue
        print(f"Downloading {url}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
