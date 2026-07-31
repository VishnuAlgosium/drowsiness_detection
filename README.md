# Driver Monitoring System (Drowsiness + Yawn + Phone Use)

Webcam-based driver monitoring. Watches for three things in one shared
capture loop and raises a distinct audio alert (plus a logged event) for
each:

| Signal      | Method                                   | Threshold(s) in `config.py`                        |
|-------------|------------------------------------------|------------------------------------------------------|
| Drowsiness  | MediaPipe FaceLandmarker + EAR           | `EAR_THRESHOLD`, `CONSEC_FRAMES`, `ALERT_COOLDOWN_SEC`|
| Yawning     | MediaPipe FaceLandmarker + MAR           | `MAR_THRESHOLD`, `YAWN_CONSEC_FRAMES`, `YAWN_COOLDOWN_SEC` |
| Phone use   | YOLO ONNX object detection               | `PHONE_CONF_THRESHOLD`, `PHONE_CONFIRM_FRAMES`, `PHONE_CONFIRM_WINDOW`, `PHONE_COOLDOWN_SEC` |

## Setup

```bash
pip install -r requirements.txt
```

Two models are required in `models/`:

- `models/face_landmarker.task` — download with:
  ```bash
  wget -O models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  ```
- `models/phone_detection_v2_ncnn_model/` — the phone-detection NCNN export folder
  (contains `model.ncnn.param`, `model.ncnn.bin`, `metadata.yaml`). Copy the whole
  folder in manually.

If you only want drowsiness/yawn, set `PHONE_DETECTION_ENABLED = False` in
`drowsy_detect/config.py` and you can skip the phone model entirely.

## Run

```bash
python main.py
```

Keys (video window must have focus): `v` toggles the display on/off, `q` quits.

## Output

- Console: `[ALERT #n]`, `[YAWN #n]`, `[PHONE #n]` lines as they happen.
- `logs/<camera_id>_<date>.jsonl` — one JSON line per alert (all three types).
- `logs/frames_<timestamp>.csv` — one row per frame: EAR, MAR, phone confidence,
  and FPS, for tuning thresholds after a QA session.

## Note on video lag

The face landmarker is lightweight and keeps up with the camera every frame.
YOLO inference is much heavier on CPU — running it every frame is what causes
the video feed to visibly lag behind real time, because frames pile up faster
than they're processed. The NCNN export is generally faster on CPU than the
ONNX/PyTorch versions, which helps, but two things fix the root cause (both
already applied here):

1. `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` — only the newest frame is kept, so
   the feed can't fall further and further behind.
2. `config.PHONE_DETECT_EVERY_N_FRAMES` — phone detection only runs every
   Nth frame (default 3), reusing the last result in between. EAR/MAR still
   run every frame since they're cheap. A phone held up to the camera stays
   in view for many frames, so this doesn't hurt detection reliability —
   only how many frames it takes to notice.

If the feed still lags on the QA machine, raise `PHONE_DETECT_EVERY_N_FRAMES`
or lower `PHONE_IMG_SIZE` (e.g. 480) in `config.py`.
