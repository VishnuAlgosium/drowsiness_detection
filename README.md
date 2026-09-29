# Driver Monitoring System

Webcam-based driver monitoring. Watches for nine things in one shared
capture loop and raises a distinct audio alert (plus a logged event) for
each:

| Signal              | Method                                 | Key thresholds in `config.py`                                  |
|----------------------|----------------------------------------|------------------------------------------------------------------|
| Drowsiness           | MediaPipe FaceLandmarker + EAR         | `EAR_THRESHOLD`, `EYES_CLOSED_HOLD_SEC`, `ALERT_COOLDOWN_SEC`     |
| Eye occlusion        | Blink-visibility monitor (no blinks seen for a while, e.g. sunglasses) | `NO_BLINK_TIMEOUT_SEC`, `MAX_BLINK_SEC` |
| PERCLOS (rolling)    | % of last window with eyes closed      | `PERCLOS_WINDOW_SEC`, `PERCLOS_ALERT_THRESHOLD`, `PERCLOS_COOLDOWN_SEC` |
| Yawning              | MediaPipe FaceLandmarker + MAR         | `MAR_THRESHOLD`, `YAWN_HOLD_SEC`, `YAWN_MAX_TRANSITIONS`, `YAWN_COOLDOWN_SEC` |
| Distraction (gaze)   | Head-pose yaw/pitch/roll vs. calibrated baseline | `YAW_ANGLE_MAX`, `PITCH_ANGLE_MAX`, `ROLL_ANGLE_MAX`, `DISTRACTION_HOLD_SEC` |
| Head drop            | Sudden downward pitch (nodding off)    | `PITCH_RATIO_DOWN`, `HEAD_DROP_DELTA`, `HEAD_DROP_HOLD_SEC`       |
| Phone use            | YOLO NCNN object detection             | `PHONE_CONF_THRESHOLD`, `PHONE_CONFIRM_WINDOW_SEC`/`RATIO`, `PHONE_COOLDOWN_SEC` |
| Cigarette use        | YOLO NCNN classification model         | `CIGARETTE_CONF_THRESHOLD`, `CIGARETTE_CONFIRM_WINDOW_SEC`/`RATIO`, `CIGARETTE_COOLDOWN_SEC` |
| Seatbelt (missing)   | YOLO NCNN detection model (alerts on absence) | `SEATBELT_CONF_THRESHOLD`, `SEATBELT_ABSENT_WINDOW_SEC`/`RATIO`, `SEATBELT_STARTUP_GRACE_SEC` |

Each detector can be toggled independently in `config.py` via its
`*_DETECTION_ENABLED` flag (phone, cigarette, seatbelt).

All hold times and confirm windows are in seconds, so behaviour doesn't
change with FPS. The YOLO detectors run on staggered frames (one model per
frame) via `*_DETECT_EVERY_N_FRAMES` and `*_DETECT_OFFSET`.

## Setup

```bash
pip install -r requirements.txt
```

Models required in `models/`:

- `models/face_landmarker.task` — download with:
  ```bash
  wget -O models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  ```
- `models/phone_detection_v6_ncnn_model/` — phone-detection NCNN export folder
  (contains `model.ncnn.param`, `model.ncnn.bin`, `metadata.yaml`).
- `models/cigarette_classification_v1_ncnn_model/` — cigarette-classification
  NCNN export folder. Classes: `cigarrete` (spelled as in the model's
  `metadata.yaml`), `nocigarette`.
- `models/seatbelt_detection_v1_ncnn_model/` — seatbelt-detection NCNN export
  folder. Class: `seatbelt` (the model detects the belt itself; an alert
  fires when it's absent for a sustained window, not when it's found).

Set the matching `*_DETECTION_ENABLED = False` in `config.py` to skip a
model entirely (and its files aren't required at startup).

## Run

```bash
python main.py
```

Keys are read from the terminal running `main.py` (not the video window):
`v` toggles the display on/off, `q` quits. The display starts off
(`DISPLAY_ON_START`).

## Output

- Console: `[ALERT #n]`, `[YAWN #n]`, `[PHONE #n]`, `[CIGARETTE #n]`,
  `[SEATBELT #n]`, `[DISTRACTION #n]`, `[HEAD DROP #n]`, `[OCCLUSION #n]`,
  `[PERCLOS #n]` lines as they happen.
- `logs/<camera_id>_<date>.jsonl` — one JSON line per alert, across all
  detectors.
- `logs/frames_<timestamp>.csv` — one row per frame: face detected, EAR,
  MAR, phone/cigarette/seatbelt confidence, PERCLOS, yaw/pitch/roll, pitch
  ratio, and FPS, for tuning thresholds after a QA session. The `fps` column
  is per-frame and spiky; compute sustained FPS from the `timestamp` column.
- Old files under `logs/` are pruned automatically at startup, per
  `LOG_RETENTION_DAYS`.