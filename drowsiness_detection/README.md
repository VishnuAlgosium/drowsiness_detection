# Drowsiness Detection

Standalone webcam driver-drowsiness detector using MediaPipe Tasks
`FaceLandmarker` and EAR (Eye Aspect Ratio) monitoring, with a pygame
beep alert and an ffplay-based low-latency display.

## Folder structure

```
drowsiness_detection/
├── main.py                    # entry point — run this
├── requirements.txt
├── models/
│   └── face_landmarker.task   # download separately (see below)
└── drowsy_detect/
    ├── __init__.py
    ├── config.py               # all tunable constants
    ├── ear.py                  # EAR math
    ├── audio.py                # pygame beep alert
    ├── display.py              # ffplay pipe-based display
    ├── keyboard_input.py       # non-blocking single-key reader
    └── detector.py             # main capture/inference/alert loop
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the face landmarker model
wget -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# ffplay is required for the live display window
sudo apt install ffmpeg
```

## Run

```bash
python main.py
```

Keyboard controls while running (in the terminal):
- `v` — toggle the ffplay display window on/off
- `q` — quit

## Tuning

Adjust detection sensitivity in `drowsy_detect/config.py`:
- `EAR_THRESHOLD` — eye-closed threshold (lower = more closed)
- `CONSEC_FRAMES` — consecutive below-threshold frames before alert
- `ALERT_COOLDOWN_SEC` — minimum seconds between repeated alerts
