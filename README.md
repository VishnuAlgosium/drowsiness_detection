# car_mobile_face_detection / drowsiness_detection

Real-time driver drowsiness detection using MediaPipe Face Mesh (EAR-based).
Display is GPU-accelerated via an `ffplay` subprocess (SDL2/OpenGL) instead of
`cv2.imshow`, so it works regardless of which OpenCV build (headless or GUI)
you have installed.

## Structure

```
car_mobile_face_detection/
├── run_drowsiness.py             # entry point
├── requirements.txt
├── README.md
└── drowsiness_detection/         # package
    ├── __init__.py
    ├── config.py                 # thresholds, colors, landmark indices
    ├── audio.py                  # pygame beep alert
    ├── ear.py                    # Eye Aspect Ratio calc
    ├── display.py                # FFplayDisplay (GPU display via ffplay)
    ├── input_keys.py             # TerminalKeyReader (non-blocking keypress)
    ├── drawing.py                 # HUD / overlay drawing helpers
    └── detector.py                # main capture + inference loop
```

## Setup

```bash
pip install -r requirements.txt
sudo apt-get install -y ffmpeg   # provides ffplay
```

## Run

```bash
python run_drowsiness.py
```

## Controls (typed into the terminal, no window focus needed)

| Key   | Action                        |
|-------|--------------------------------|
| q     | quit                           |
| + / - | adjust EAR threshold           |
| [ / ] | adjust consecutive-frame limit |
| r     | reset counters                 |

Closing the ffplay window also exits the program.
