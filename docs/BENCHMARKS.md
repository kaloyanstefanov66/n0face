# Benchmarks

The aim is measurable claims, including the failures. Every run appends a row to
`bench/results/perf.md` and saves full JSON next to it. Commit both, good and bad.

## 1. Performance and detection failures (`bench/perf.py`)

```bash
python bench/perf.py --source <video or webcam index> --mode anonymize --expect-faces 1 --label "<condition>"
```

The benchmark runs the same detect -> mask -> effect path the app uses and reports:

| Metric | Meaning |
|---|---|
| `throughput_fps` | Frames processed per second (camera wait excluded for video files) |
| `latency_ms.detect / mask_and_effect / total` | p50, p95 and max per stage |
| `frames_with_no_detection`, `longest_miss_streak_frames` | Raw detector misses |
| `fail_closed_frames` | Frames where the whole image was hidden |
| `potential_leak_frames` | Frames with fewer detections than `--expect-faces` that were **not** hidden. Upper bound on frames that may have shown a face. |
| `cpu_percent`, `peak_rss_mb` | Process resource use (needs `psutil`) |

### Conditions to cover

Record the same scripted clip (about 30 s, consenting participants only) under each condition:

1. Daylight, frontal, still
2. Low light (a single lamp)
3. Strong backlight (a window behind you)
4. Head turns to ±60° yaw and nodding
5. Fast motion
6. Partial occlusion (hand, glasses, mask, hair)
7. Two or more faces, including one entering mid-clip
8. 480p, 720p and 1080p input

### Results

No published results yet. The first runs on real recordings go here, with machine specs.

## 2. Anonymization effectiveness (planned)

Question: **can a face-recognition model still match a person after n0face processes them?**

Method:

1. **Data:** a consented set of N identities, each with a clean enrollment photo and video clips
   under the conditions above. Public sets (for example LFW, or YouTube Faces for video) can
   supplement it, within their licenses.
2. **Processing:** run every clip through each mode/effect combination.
3. **Recognizers:** at least three different families, such as an ArcFace model (InsightFace),
   FaceNet and dlib ResNet. Use the face detector each recognizer ships with.
4. **Metrics per mode, effect and condition:**
   - Detection rate: does the recognizer's own detector still find a face?
   - Verification: true-accept rate at a fixed false-accept rate (for example 1e-3), comparing
     processed frames against the clean enrollment photo.
   - Identification: rank-1 accuracy against a gallery of all identities.
   - Baseline: the same metrics on unprocessed clips.
5. **Report:** full tables, including the combinations where matching still succeeds. Expect
   `mask` and `pixelate` to leak. That's the point of measuring.

Out of scope for this test, and to be stated with any results: re-identification from body,
clothing, voice, background or behavior.
