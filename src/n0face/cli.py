"""Terminal entry point. The desktop app is `n0face`; this is `n0face-cli`.

    n0face-cli process photo.jpg clip.mp4 holiday/ --mode anonymize
    n0face-cli --preview                       # live webcam -> virtual camera
"""

import argparse
import sys
from pathlib import Path

import cv2

from n0face import plugins
from n0face.effects import effect_names
from n0face.modes import MODE_NAMES
from n0face.theme import THEMES

PREVIEW_WINDOW = "n0face (q to quit)"


def parse_args(argv=None) -> argparse.Namespace:
    plugins.load()
    p = argparse.ArgumentParser(
        prog="n0face-cli",
        description="Mask or anonymize your face and publish it as a virtual webcam. "
        "Everything runs on this machine; nothing is recorded or sent anywhere. "
        "For photos and videos, use: n0face-cli process --help",
    )
    p.add_argument("--mode", choices=MODE_NAMES, default="mask", help="privacy mode (default: mask)")
    p.add_argument("--effect", choices=effect_names(), help="effect within the mode (default: mode's first)")
    p.add_argument("--theme", choices=list(THEMES), default="cyan")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-faces", type=int, default=4)
    p.add_argument("--preview", action="store_true", help="show a local preview window with hotkeys")
    p.add_argument("--no-virtualcam", action="store_true", help="don't publish a virtual webcam")
    p.add_argument("--no-hud", action="store_true", help="disable the HUD overlay in mask mode")
    return p.parse_args(argv)


def parse_process_args(argv) -> argparse.Namespace:
    plugins.load()
    p = argparse.ArgumentParser(
        prog="n0face-cli process",
        description="Mask or anonymize faces in existing photos and videos. Originals are never modified; "
        "outputs are written as <name>.n0face.<ext> with all metadata removed.",
    )
    p.add_argument("inputs", nargs="+", type=Path, help="image/video files or folders")
    p.add_argument("--out", type=Path, help="output folder (default: next to each original)")
    p.add_argument("--mode", choices=MODE_NAMES, default="anonymize", help="privacy mode (default: anonymize)")
    p.add_argument("--effect", choices=effect_names(), help="effect within the mode (default: mode's first)")
    p.add_argument("--theme", choices=list(THEMES), default="cyan")
    p.add_argument("--no-hud", action="store_true", help="disable the HUD overlay in mask mode")
    p.add_argument(
        "--on-miss", choices=["hide", "keep"], default="hide",
        help="when no face is found in a photo/frame: hide it entirely (default) or keep it unmodified",
    )
    p.add_argument("--window", type=int, default=4, help="video: frames on each side whose detections also cover a frame")
    p.add_argument("--fast", action="store_true", help="skip tiled detection (faster, misses more small faces)")
    p.add_argument("--keep-audio", action="store_true", help="video: keep the original audio (voices are NOT anonymized)")
    return p.parse_args(argv)


def process_main(argv) -> None:
    args = parse_process_args(argv)
    from n0face.detector import StillDetector
    from n0face.media import Cancelled, MediaProcessor, collect_inputs, ffmpeg_path
    from n0face.modes import MODES
    from n0face.netguard import lock_network

    files = collect_inputs(args.inputs)
    if not files:
        raise SystemExit("No supported images or videos found.")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    if args.keep_audio and ffmpeg_path() is None:
        print("[warn] ffmpeg not found: audio can't be kept")

    detector = StillDetector(thorough=not args.fast)
    lock_network()
    print(f"[privacy] network disabled; mode {args.mode}: {MODES[args.mode].claim}")

    def progress(frac, _frame):
        print(f"\r  {frac * 100:5.1f}%", end="", flush=True)

    processor = MediaProcessor(
        detector, args.mode, args.effect, THEMES[args.theme], hud=not args.no_hud,
        on_miss=args.on_miss, window=args.window, keep_audio=args.keep_audio, progress=progress,
    )
    failures = 0
    try:
        for i, src in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {src}")
            try:
                report = processor.process(src, args.out)
            except (RuntimeError, OSError, ValueError) as e:
                failures += 1
                print(f"\r  FAILED: {e}")
                continue
            print(f"\r  {report.summary()}")
            for w in report.warnings:
                print(f"  [warn] {w}")
            if report.kept_unmasked_frames:
                print("  [warn] some content was kept UNMASKED because no face was found (--on-miss keep)")
    except (KeyboardInterrupt, Cancelled):
        print("\n[cancelled]")
    finally:
        detector.close()
    sys.exit(1 if failures else 0)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "process":
        return process_main(argv[1:])
    args = parse_args(argv)
    if args.no_virtualcam and not args.preview:
        raise SystemExit("Nothing to output: use --preview and/or drop --no-virtualcam.")

    from n0face.capture import Camera
    from n0face.detector import FaceDetector
    from n0face.modes import MODES
    from n0face.netguard import lock_network
    from n0face.output import VirtualCam
    from n0face.pipeline import Engine

    # All imports and the model are local; from here on Python-level networking is off.
    lock_network()
    print("[privacy] network disabled for this process; no frames are recorded or saved")

    camera = Camera(args.camera, args.width, args.height, args.fps)
    print(f"[camera] LIVE: camera {args.camera} is on. It turns off when n0face exits.")
    detector = FaceDetector(max_faces=args.max_faces)
    engine = Engine(detector, args.mode, args.effect, THEMES[args.theme], hud=not args.no_hud)
    sink = None
    try:
        if not args.no_virtualcam:
            sink = VirtualCam(*camera.size, args.fps)
            print(f"[stream] publishing to virtual camera: {sink.device}")
        print(f"[mode] {engine.mode_name}/{engine.effect_name}: {MODES[engine.mode_name].claim}")
        if args.preview:
            print("[keys] m cycles mode, 1-3 switch effect, q quits")
        while True:
            safe = engine.process(camera.read())
            if sink:
                sink.send(safe)
            if not args.preview:
                continue
            cv2.imshow(PREVIEW_WINDOW, safe)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                engine.set_mode(MODE_NAMES[(MODE_NAMES.index(engine.mode_name) + 1) % len(MODE_NAMES)])
                print(f"[mode] {engine.mode_name}/{engine.effect_name}: {engine.mode.claim}")
            elif ord("1") <= key < ord("1") + len(engine.mode.effects):
                engine.set_effect(engine.mode.effects[key - ord("1")])
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        print("[camera] off")
        detector.close()
        if sink:
            sink.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
