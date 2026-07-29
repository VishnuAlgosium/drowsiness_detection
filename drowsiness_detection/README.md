# Drowsiness Detection

Standalone webcam driver-drowsiness detector using MediaPipe Tasks
`FaceLandmarker` and EAR (Eye Aspect Ratio) monitoring, with a pygame
beep alert and an ffplay-based low-latency display.

## Folder structure

```
drowsiness_detection/
├── main.py                    # entry point — run this
├── requirements.txt
├── README.md
├── models/
│   └── face_landmarker.task   # download separately (see below)
└── drowsy_detect/
    ├── __init__.py
    ├── config.py               # all tunable constants
    ├── ear.py                  # EAR math
    ├── audio.py                # pygame beep alert (non-blocking)
    ├── display.py              # ffplay pipe-based display
    ├── keyboard_input.py       # non-blocking single-key reader
    └── detector.py             # main capture/inference/alert loop
```

Run everything from **this** directory — `main.py` must sit next to
`drowsy_detect/` for the import in `main.py` to resolve.

## Requirements

Tested on a Raspberry Pi with Python 3.13.5. Install with:

```bash
pip install -r requirements.txt
```

| Package        | Version   | Purpose                              |
|----------------|-----------|---------------------------------------|
| opencv-python  | 5.0.0.93  | Camera capture, frame drawing/resize  |
| numpy          | 2.5.1     | Beep waveform + landmark math         |
| mediapipe      | 1.0.0     | FaceLandmarker (face mesh landmarks)  |
| scipy          | 1.18.0    | Euclidean distance for EAR calc       |
| pygame         | 2.6.1     | Alert beep playback                   |

System dependency (not installed via pip):

```bash
sudo apt install ffmpeg   # provides ffplay, used for the display window
```

## Model file

The face landmark model isn't bundled — download it into `models/`:

```bash
wget -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Run

```bash
python main.py
```

Keyboard controls while running (in the terminal running the script):
- `v` — toggle the ffplay display window on/off
- `q` — quit

## Tuning

Adjust detection sensitivity in `drowsy_detect/config.py`:
- `EAR_THRESHOLD` — eye-closed threshold (lower = more closed)
- `CONSEC_FRAMES` — consecutive below-threshold frames before alert
- `ALERT_COOLDOWN_SEC` — minimum seconds between repeated alerts

## Known fixes applied

- `audio.py`'s alert used to busy-wait (`while pygame.mixer.get_busy(): ...`)
  until the beep finished, which froze the video feed for ~1 second on every
  alert. It's now fire-and-forget — `sound.play()` hands off to SDL's audio
  thread and the capture/display loop keeps running uninterrupted.
