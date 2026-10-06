<div align="center">

# 🚘 **Driver Monitoring System (DMS)**

### 🚨 Real-Time Driver Drowsiness, Distraction & Identity Verification Pipeline

A **production-grade AI monitoring pipeline** built for real-time video stream & webcam surveillance. Combines **MediaPipe Face Mesh + MobileFaceNet INT8 + YOLO NCNN Edge Models** for multi-signal driver safety and biometric identification with ultra-low latency.

> ⚙️ Powered by **MediaPipe + YOLO NCNN + MobileFaceNet INT8**  
> 🧠 Engineered for **low false-positive rates & balanced CPU load**  
> 🧩 Part of the **CampNeuron AI Series** — engineered by the **Algosium AI Team**

---

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](#)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarks-0097A7?logo=google&logoColor=white)](#)
[![YOLO](https://img.shields.io/badge/YOLO-NCNN%20Export-00599C?logo=ultralytics&logoColor=white)](#)
[![Conda](https://img.shields.io/badge/Conda-drowsinessVenv-44A833?logo=anaconda&logoColor=white)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20|%20x86__64-lightgrey?logo=linux&logoColor=white)](#)

</div>

---

## 📖 Overview

The **Driver Monitoring System (DMS)** delivers a comprehensive real-time safety telemetry solution inside vehicle cabins or industrial monitoring setups. Built with performance-first design principles, the engine tracks driver states, behavioral anomalies, safety non-compliance, and biometric identity using a single shared frame-processing loop to prevent CPU bottlenecks.

---

## 🚀 Key Features & Detection Signals

The system actively evaluates **9 simultaneous safety signals** alongside biometric identity verification:

| Signal | Detection Method | Key Configuration (`src/config.py`) |
| :--- | :--- | :--- |
| 👤 **Driver Identification** | MobileFaceNet INT8 + MediaPipe Alignment | `FACE_RECOGNITION_ENABLED`, `FACE_RECOGNITION_EVERY_N_FRAMES` |
| 👁️ **Drowsiness (EAR)** | MediaPipe FaceLandmarker + Eye Aspect Ratio | `EAR_THRESHOLD`, `EAR_THRESHOLD_RATIO`, `CONSEC_FRAMES` |
| 🕶️ **Eye Occlusion** | Blink-visibility monitor (detects IR glasses/blocking) | `NO_BLINK_TIMEOUT_SEC`, `MAX_BLINK_FRAMES` |
| 📊 **PERCLOS (Rolling)** | % Eye closure over trailing temporal window | `PERCLOS_WINDOW_SEC`, `PERCLOS_ALERT_THRESHOLD` |
| 🥱 **Yawning (MAR)** | Mouth Aspect Ratio + Mouth Corner Lift | `MAR_THRESHOLD`, `YAWN_HOLD_SEC`, `YAWN_MAX_TRANSITIONS` |
| 🗣️ **Distraction (Gaze)** | Head Pose (Yaw / Pitch / Roll) vs Calibrated Baseline | `YAW_ANGLE_MAX`, `PITCH_ANGLE_MAX`, `ROLL_ANGLE_MAX` |
| 📉 **Head Drop** | Sudden downward head pitch movement (nodding off) | `PITCH_RATIO_DOWN`, `HEAD_DROP_DELTA` |
| 📱 **Phone Usage** | YOLO NCNN Object Detection | `PHONE_CONF_THRESHOLD`, `PHONE_CONFIRM_FRAMES` |
| 🚬 **Cigarette Usage** | YOLO NCNN Classification Model | `CIGARETTE_CONF_THRESHOLD`, `CIGARETTE_CONFIRM_FRAMES` |
| 🎗️ **Seatbelt Absence** | YOLO NCNN Detection Model (Sustained Absence) | `SEATBELT_CONF_THRESHOLD`, `SEATBELT_ABSENT_FRAMES` |

> 💡 **Modular Control:** Each detector can be toggled on/off independently in `src/config.py` using `*_DETECTION_ENABLED` flags.

---

## 🏗️ Performance & Architecture

- **Shared Frame Pipeline:** MediaPipe face landmarks are calculated once per video frame and shared across drowsiness, yawn, gaze, head drop, eye occlusion, PERCLOS, cigarette cropping, and face recognition.
- **CPU Load Balancing:** Heavy inference routines (Phone, Cigarette, Seatbelt, Face Recognition) run on staggered frame intervals (`*_DETECT_EVERY_N_FRAMES` and `*_DETECT_OFFSET`) to evenly spread compute load across frames.
- **Asynchronous Frame Grabber:** `FrameGrabber` executes camera capture on a dedicated thread to eliminate I/O blocking and input frame buffering lag.
- **Pre-computed Embedding DB:** Driver face embeddings (MobileFaceNet INT8) are cached in `models/employee_db.npz`. At startup, an interactive 3-second prompt lets you load cached embeddings instantly or re-index from `employee/`.

---

## 🛠️ Environment Setup & Installation

### 1. Conda Environment Setup (`drowsinessVenv`)

We recommend using **Conda** for isolated dependency management.

```bash
# Create the conda environment
conda create -n drowsinessVenv python=3.11 -y

# Activate the environment
conda activate drowsinessVenv

# Clone the repository (if not already local)
git clone https://github.com/VishnuAlgosium/drowsiness_detection.git
cd drowsiness_detection

# Install required Python dependencies
pip install -r requirements.txt
```

---

## 📦 Model Assets Preparation

Ensure the following pre-trained models are present in the `models/` directory:

```text
models/
├── face_landmarker.task
├── mobilefacenet_int8.tflite
├── phone_detection_v6_ncnn_model/
│   ├── model.ncnn.param
│   └── model.ncnn.bin
├── cigarette_classification_v1_ncnn_model/
│   ├── model.ncnn.param
│   └── model.ncnn.bin
└── seatbelt_detection_v1_ncnn_model/
    ├── model.ncnn.param
    └── model.ncnn.bin
```

### Download Google MediaPipe Face Landmarker
```bash
wget -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## 👤 Employee Dataset & Driver Registration

To register authorized drivers for automatic identity recognition, place clear facial photos into the `employee/` directory using the following file naming convention (`ID_Name.jpg` or `ID.jpg`):

```text
employee/
├── EMP001_John_Doe.jpg
├── EMP002_Jane_Smith.jpg
└── EMP003_Robert.png
```

During startup, face alignment (112x112) and feature extraction generate embeddings cached to `models/employee_db.npz` for near-instant boot times.

---

## 🎮 Running the Application

Ensure your conda environment is activated:

```bash
conda activate drowsinessVenv
python main.py
```

### 🔁 Startup Database Prompt
If `models/employee_db.npz` exists, the system presents an interactive options prompt:
```text
[QUESTION] Found cached employee database. Do you want to update/rebuild from 'employee' folder? (y/N):
```
- Press **`y`** to rebuild embeddings from the `employee/` folder.
- Press **`Enter`** (or wait 3 seconds) to proceed with cached embeddings.

### ⌨️ Runtime Controls
*(Focus must be on the OpenCV video display window)*
- **`v`**: Toggle display rendering on/off (useful for headless/background runs).
- **`q`**: Gracefully stop the application and release camera resources.

---

## 📊 Logging & Audit Telemetry

The system provides robust real-time diagnostic reporting and file logging:

- 📺 **Live Console Alerts:** Event tags printed directly (`[ALERT #n]`, `[YAWN #n]`, `[PHONE #n]`, `[CIGARETTE #n]`, `[SEATBELT #n]`, `[DISTRACTION #n]`, `[HEAD DROP #n]`, `[OCCLUSION #n]`, `[PERCLOS #n]`).
- 📝 **Structured JSONL Audit Logs (`logs/<camera_id>_<date>.jsonl`):** Log entries capturing every triggered alert event with timestamp & metadata.
- 📈 **Per-Frame Metrics CSV (`logs/frames_<timestamp>.csv`):** Per-frame telemetry (EAR, MAR, PERCLOS, Gaze Angles, Pitch Ratio, Confidences, and FPS) for offline analysis, model validation, and QA threshold tuning.
- 🧹 **Automatic Pruning:** Log retention automatically cleans files older than `LOG_RETENTION_DAYS` (default: 14 days) at startup.

---

## 🛠️ Advanced Model Verification Utilities

To benchmark NCNN exported models against PyTorch (`.pt`) source weights:

```bash
conda activate drowsinessVenv
python compare.py \
    --images-dir ./test_images \
    --pt-model ./models/cigarette_classification_v1.pt \
    --ncnn-model ./models/cigarette_classification_v1_ncnn_model \
    --face-model ./models/face_landmarker.task \
    --padding 0.3
```

---

## 🧩 Project Structure

```text
drowsiness_detection/
├── main.py                  # Application entry point
├── compare.py               # NCNN vs PyTorch model comparison script
├── requirements.txt         # Dependencies list
├── employee/                # Driver face dataset folder
├── models/                  # NCNN & MediaPipe model artifacts directory
├── logs/                    # Automated JSONL & CSV event telemetry logs
└── src/
    ├── config.py            # Global thresholds, timing & detection flags
    ├── detector.py          # Main processing loop & coordinator
    ├── face_identifier.py   # MobileFaceNet alignment & embedding matcher
    ├── capture.py           # Threaded FrameGrabber
    ├── ear.py               # Eye Aspect Ratio calculation
    ├── mar.py               # Mouth Aspect Ratio calculation
    ├── gaze.py              # Head pose orientation estimator
    ├── perclos.py           # PERCLOS rolling buffer computation
    ├── blink_visibility.py  # Occlusion & blink detector
    ├── phone.py             # Phone usage detection handler
    ├── cigarette.py         # Cigarette usage classification handler
    ├── seatbelt.py          # Seatbelt compliance detection handler
    ├── audio.py             # Sound alert playback synthesizer
    └── display.py           # OpenCV frame annotation & HUD overlay
```

---

<div align="center">

**Engineered with ❤️ by the Algosium AI Team**

</div>