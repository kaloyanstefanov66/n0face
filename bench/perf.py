"""Performance and detection-failure benchmark.

Runs the exact publish path (detect -> mask -> effect) over a video file or webcam and reports
per-stage latency, throughput, detection misses, fail-closed events and resource use.
Results go to bench/results/ as JSON plus a Markdown row, including the bad numbers.

    python bench/perf.py --source clip.mp4 --expect-faces 1 --mode anonymize
    python bench/perf.py --source 0 --frames 300
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from n0face import __version__
from n0face.capture import Camera
from n0face.detector import FaceDetector
from n0face.effects import make_effects
from n0face.modes import MODE_NAMES, MODES
from n0face.netguard import lock_network
from n0face.pipeline import anonymize, make_tracker

RESULTS = Path(__file__).resolve().parent / "results"


def pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="video file path or webcam index")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = whole file / 300 for webcam)")
    ap.add_argument("--mode", choices=MODE_NAMES, default="anonymize")
    ap.add_argument("--effect", help="effect within the mode (default: mode's first)")
    ap.add_argument("--expect-faces", type=int, default=None, help="faces truly present in every frame, if known")
    ap.add_argument("--label", default="", help="free-text condition label, e.g. 'low light, 2 faces'")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    limit = args.frames or (300 if isinstance(source, int) else 10**9)

    try:
        import psutil

        proc = psutil.Process()
        proc.cpu_percent(None)
    except ImportError:
        proc = None

    lock_network()
    camera = Camera(source)
    resolution = list(camera.size)
    detector = FaceDetector()
    mode = MODES[args.mode]
    effect_name = args.effect if args.effect in mode.effects else mode.effects[0]
    effect = make_effects(detector.edges)[effect_name]
    tracker = make_tracker(mode)

    detect_ms, render_ms, total_ms = [], [], []
    face_counts, lost_flags = [], []
    peak_rss = 0
    start = time.perf_counter()
    try:
        while len(total_ms) < limit:
            try:
                frame = camera.read()
            except EOFError:
                break
            t0 = time.perf_counter()
            faces = detector.detect(frame, int((t0 - start) * 1000))
            t1 = time.perf_counter()
            _, lost = anonymize(frame, faces, tracker, mode, effect, t1 - start)
            t2 = time.perf_counter()
            detect_ms.append((t1 - t0) * 1000)
            render_ms.append((t2 - t1) * 1000)
            total_ms.append((t2 - t0) * 1000)
            face_counts.append(len(faces))
            lost_flags.append(lost)
            if proc is not None and len(total_ms) % 30 == 0:
                peak_rss = max(peak_rss, proc.memory_info().rss)
    finally:
        wall = time.perf_counter() - start
        camera.close()
        detector.close()

    n = len(total_ms)
    if n == 0:
        sys.exit("No frames processed")

    misses = [c == 0 for c in face_counts]
    streak = longest = 0
    for m in misses:
        streak = streak + 1 if m else 0
        longest = max(longest, streak)

    result = {
        "version": __version__,
        "label": args.label,
        "source": str(args.source),
        "mode": mode.name,
        "effect": effect_name,
        "frames": n,
        "resolution": resolution,
        "throughput_fps": n / wall,
        "latency_ms": {
            stage: {"p50": pct(v, 50), "p95": pct(v, 95), "max": max(v)}
            for stage, v in (("detect", detect_ms), ("mask_and_effect", render_ms), ("total", total_ms))
        },
        "frames_with_no_detection": sum(misses),
        "longest_miss_streak_frames": longest,
        "fail_closed_frames": sum(lost_flags),
        "cpu_percent": proc.cpu_percent(None) if proc else None,
        "peak_rss_mb": peak_rss / 1e6 if proc else None,
        "machine": {"platform": platform.platform(), "cpu": platform.processor(), "python": platform.python_version()},
    }
    if args.expect_faces is not None:
        # A frame where fewer faces were detected than exist, and the frame was NOT hidden,
        # may have shown a face. Without per-pixel ground truth this is an upper bound.
        result["expected_faces"] = args.expect_faces
        result["potential_leak_frames"] = sum(
            1 for c, lost in zip(face_counts, lost_flags) if c < args.expect_faces and not lost
        )
        if args.expect_faces:
            result["detection_recall"] = statistics.fmean(min(c, args.expect_faces) / args.expect_faces for c in face_counts)

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"perf-{stamp}.json").write_text(json.dumps(result, indent=2))
    lat = result["latency_ms"]
    row = (
        f"| {stamp} | {args.label or '-'} | {mode.name}/{effect_name} | {n} | {result['throughput_fps']:.1f} | "
        f"{lat['detect']['p50']:.1f} / {lat['detect']['p95']:.1f} | {lat['total']['p50']:.1f} / {lat['total']['p95']:.1f} | "
        f"{result['frames_with_no_detection']} (max streak {longest}) | {result['fail_closed_frames']} | "
        f"{result.get('potential_leak_frames', '-')} |\n"
    )
    table = RESULTS / "perf.md"
    if not table.exists():
        table.write_text(
            "| run | condition | mode/effect | frames | fps | detect p50/p95 ms | total p50/p95 ms | "
            "no-detection frames | fail-closed frames | potential leak frames |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n"
        )
    with table.open("a") as f:
        f.write(row)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
