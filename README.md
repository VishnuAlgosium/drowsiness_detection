# Driver Monitoring System

Webcam-based driver monitoring. Watches for five things in one shared
capture loop and raises a distinct audio alert (plus a logged event) for
each:

| Signal              | Method                                 | Key thresholds in `config.py`                                  |
|----------------------|----------------------------------------|------------------------------------------------------------------|
| Drowsiness           | MediaPipe FaceLandmarker + EAR         | `EAR_THRESHOLD`, `CONSEC_FRAMES`, `ALERT_COOLDOWN_SEC`            |
| Eye occlusion        | Blink-visibility monitor (no blinks seen for a while, e.g. sunglasses) | `NO_BLINK_TIMEOUT_SEC`, `MAX_BLINK_FRAMES` |
| PERCLOS (rolling)    | % of last window with eyes closed      | `PERCLOS_WINDOW_SEC`, `PERCLOS_ALERT_THRESHOLD`, `PERCLOS_COOLDOWN_SEC` |
| Yawning              | MediaPipe FaceLandmarker + MAR         | `MAR_THRESHOLD`, `YAWN_HOLD_SEC`, `YAWN_MAX_TRANSITIONS`, `YAWN_COOLDOWN_SEC` |
| Distraction (gaze)   | Head-pose yaw/pitch/roll vs. calibrated baseline | `YAW_ANGLE_MAX`, `PITCH_ANGLE_MAX`, `ROLL_ANGLE_MAX`, `DISTRACTION_HOLD_SEC` |
| Head drop            | Sudden downward pitch (nodding off)    | `PITCH_RATIO_DOWN`, `HEAD_DROP_DELTA`, `HEAD_DROP_HOLD_FRAMES`    |
| Phone use            | YOLO NCNN object detection             | `PHONE_CONF_THRESHOLD`, `PHONE_CONFIRM_FRAMES`/`WINDOW`, `PHONE_COOLDOWN_SEC` |
| Cigarette use        | YOLO NCNN classification model         | `CIGARETTE_CONF_THRESHOLD`, `CIGARETTE_CONFIRM_FRAMES`/`WINDOW`, `CIGARETTE_COOLDOWN_SEC` |
| Seatbelt (missing)   | YOLO NCNN detection model (alerts on absence) | `SEATBELT_CONF_THRESHOLD`, `SEATBELT_ABSENT_FRAMES`/`WINDOW`, `SEATBELT_STARTUP_GRACE_SEC` |

Each detector can be toggled independently in `config.py` via its
`*_DETECTION_ENABLED` flag (phone, cigarette, seatbelt).

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
- `models/phone_detection_v4_ncnn_model/` — phone-detection NCNN export folder
  (contains `model.ncnn.param`, `model.ncnn.bin`, `metadata.yaml`).
- `models/cigarette_detection_v3_ncnn_model/` — cigarette-classification NCNN
  export folder. Classes: `cigarette`, `nocigarette`.
- `models/seatbelt_detection_ncnn_model/` — seatbelt-detection NCNN export
  folder. Class: `seatbelt` (the model detects the belt itself; an alert
  fires when it's absent for a sustained window, not when it's found).

Set the matching `*_DETECTION_ENABLED = False` in `config.py` to skip a
model entirely (and its files aren't required at startup).

## Run

```bash
python main.py
```

Keys (video window must have focus): `v` toggles the display on/off, `q` quits.

## Output

- Console: `[ALERT #n]`, `[YAWN #n]`, `[PHONE #n]`, `[CIGARETTE #n]`,
  `[SEATBELT #n]`, `[DISTRACTION #n]`, `[HEAD DROP #n]`, `[OCCLUSION #n]`,
  `[PERCLOS #n]` lines as they happen.
- `logs/<camera_id>_<date>.jsonl` — one JSON line per alert, across all
  detectors.
- `logs/frames_<timestamp>.csv` — one row per frame: EAR, MAR, phone/
  cigarette/seatbelt confidence, PERCLOS, yaw/pitch/roll, pitch ratio, and
  FPS, for tuning thresholds after a QA session.
- Old files under `logs/` are pruned automatically at startup, per
  `LOG_RETENTION_DAYS`.