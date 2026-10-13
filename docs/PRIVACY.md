# Privacy design and data flow

## Data flow

```
webcam ──frames──> [capture]  (RAM only)
                      │
                      ▼
                 [detector]  MediaPipe FaceLandmarker + PoseLandmarker, local model files, CPU
                      │ landmarks (478 points per face) + heads the face model missed
                      ▼
                 [mask + tracker]  expanded hull/ellipse, OR'd with previous frame,
                      │            fail-closed after N missed frames
                      ▼
                 [effect]  generated fill (no camera pixels, except "pixelate" and mask mode's blur)
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
   [virtual camera]          [app preview]
   OBS Virtual Camera        shows the same processed frame
   driver (shared memory)
          │
          ▼
   Discord / Teams / Zoom / OBS ...   ← n0face's guarantees end here
```

Nothing is written to disk. Frames exist only in process memory and are overwritten every frame.
The raw camera frame never reaches the virtual camera, and the app preview never shows it.

### Photos and videos

```
original file ──read──> [detector: full image + overlapping tiles]
   (never modified)          │
                             ▼
                 [mask = union of detections in frames t-4 .. t+4]   (videos)
                             │
                             ▼
                 [effect] ──> new file  name.n0face.<ext>
                              pixels only: no EXIF/GPS/tags; video audio removed unless asked
```

- Photos are encoded by OpenCV, which writes no metadata. Videos are piped as raw frames into
  ffmpeg with `-map_metadata -1 -map_chapters -1` and bitexact flags.
- Photo orientation is applied to the pixels before processing, so stripping EXIF doesn't
  rotate the result.
- `--keep-audio` copies the original audio track. **Voices are not anonymized.**
- Frames are never written to temporary files. They go straight from memory into the encoder.
  A cancelled or failed job deletes its partial output.

## What n0face guarantees

| Claim | How it's enforced | Verified by |
|---|---|---|
| No network access at runtime | `netguard.lock_network()` blocks Python IP sockets and DNS after the model loads | `tests/test_netguard.py` |
| No recording of the live camera | The live path (`pipeline.Engine`, `app/worker.Worker`, `cli` live mode) has no file output. All file writing lives in `media.py`, used only for files the user selects. Benchmarks write only aggregate numbers. | Code review: `grep -rn "imwrite\|imencode\|VideoWriter\|tofile" src/` matches only `media.py` |
| Masked pixels don't come from your face | `Effect.render()` never receives the camera frame | `tests/test_effects.py::test_masked_region_is_independent_of_camera_pixels` |
| Fail-closed on tracking loss | `MaskTracker` hides the whole frame after `hold_frames` misses; sessions start hidden | `tests/test_mask.py`, `tests/test_modes.py` |
| Fail-closed never uses camera pixels | A lost face always falls back to static, even when pixelate is selected | `tests/test_modes.py::test_lost_face_never_uses_camera_pixels` |
| Anonymize hides face shape | The ellipse is built from a grid-snapped bounding box | `tests/test_modes.py::test_ellipse_silhouette_hides_face_shape` |
| Output files carry no metadata | OpenCV encoding for photos; ffmpeg `-map_metadata -1` for video | `tests/test_media.py::test_photo_masked_metadata_stripped_original_untouched`, `test_video_strips_metadata_and_audio_by_default` |
| Originals are never modified | Outputs get a `.n0face` name and never overwrite existing files | `tests/test_media.py::test_output_path_never_overwrites` |
| Short detector misses in videos stay covered | Mask is the union of detections over ±`window` frames | `tests/test_media.py::test_video_window_covers_short_detector_misses` |
| Heads the face model misses stay covered | The pose model finds each person's head. One with no face mesh gets a whole-head ellipse in obfuscate/anonymize; in mask mode the design is drawn on a face mesh fitted to the head, fully covered even with 3D wrap (SOLID backing) | `tests/test_heads.py` |
| A face hit on a hand doesn't stand in for the real face | Face hits on a person's hand and in no head are dropped, so they can't stop a frame failing closed | `tests/test_heads.py::test_face_on_a_hand_is_dropped` |
| Camera state is visible | The app shows a blinking CAMERA LIVE status; the camera is released on DISENGAGE or exit | Manual |

## Known limits (read these)

- **The network guard is Python-level.** Native code that calls the OS socket API directly
  bypasses it. For a hard guarantee, add an OS firewall rule that blocks the n0face/python
  executable.
- **Detection can miss.** Extreme angles, occlusion, low light and motion blur can cause misses.
  The pose model covers most face-model misses (looking down or away) with a head mask; short
  misses of both are covered by holding a growing mask; long ones hide the frame. A head
  **neither** model saw while another person was detected is not covered. See the
  "potential leak frames" metric in the benchmarks.
- **Mask mode is decorative.** It reproduces your facial geometry and expressions. Its default
  BLUR backing shows a heavy blur of your face (colour, rough shape) under the design; SOLID
  shows none. Fail-closed frames are never blurred: they're hidden like in every mode.
- **Pixelation is weak.** It is derived from your face pixels and can sometimes be matched or
  partially reversed.
- **Downstream apps are out of scope.** Once a frame reaches Discord, Teams and so on, their
  policies apply. n0face only controls what it sends.
- **Not hidden in any mode:** body, clothing, hair outside the cover, voice, room, other people,
  and anything visible on screen.

## Permissions

- **Camera:** opened only while engaged. Choose which device in the app.
- **Virtual camera:** optional. Toggle it off to use the preview only.
- **Network:** none needed after the one-time model download.
- **Disk:** reads the model file. Writes nothing (benchmarks write aggregate result files only).
