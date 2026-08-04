"""
config.py
---------
All tunable constants for the detector live here so the rest of the
codebase never hardcodes a magic number.
"""

import os

# ─────────────────────────────────────────────
# CPU / threading
# ─────────────────────────────────────────────

CUDA_VISIBLE_DEVICES = "-1"
OMP_NUM_THREADS = "4"
OPENCV_NUM_THREADS = 4
OPENCV_USE_OPENCL = False


def apply_cpu_settings() -> None:
    """Apply the CPU/threading env vars. Call this before importing cv2/mediapipe/ultralytics."""
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS


# ─────────────────────────────────────────────
# Drowsiness detection (Eye Aspect Ratio)
# ─────────────────────────────────────────────

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
ALERT_COOLDOWN_SEC = 4.0

# Startup calibration: baseline_ear * EAR_THRESHOLD_RATIO replaces the fixed
# threshold above, so eye shape doesn't need one global cutoff. 0 = skip.
EAR_CALIBRATION_FRAMES = 60
EAR_THRESHOLD_RATIO = 0.75

# EMA smoothing factor for EAR (lower = smoother, damps rubbing/occlusion noise).
EAR_SMOOTHING_ALPHA = 0.4

# Max |left_ear - right_ear| to still count as "both eyes closed" (rejects winks).
EAR_ASYMMETRY_MAX = 0.12

# Consecutive no-face frames tolerated before counters start decaying.
NO_FACE_GRACE_FRAMES = 5

# ─────────────────────────────────────────────
# Yawn detection (Mouth Aspect Ratio)
# ─────────────────────────────────────────────

MAR_THRESHOLD = 0.6          # mouth-open ratio above which counts as "open"
YAWN_CONSEC_FRAMES = 20      # ~0.6-0.7s at 30fps held open before it's a yawn
YAWN_COOLDOWN_SEC = 4.0

# Looser threshold for small/suppressed yawns, held longer to compensate.
MAR_LOW_THRESHOLD = 0.45
MAR_LOW_CONSEC_FRAMES = 35

# Oscillation filter: a yawn is one open->hold->close; talking/laughing/
# singing repeatedly opens and closes. Reject if rising edges exceed this
# within the trailing window.
YAWN_TRANSITION_WINDOW = 30
YAWN_MAX_TRANSITIONS = 1

# MediaPipe face mesh mouth landmark indices:
# top inner lip, bottom inner lip, left corner, right corner
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# ─────────────────────────────────────────────
# Phone-use detection (YOLO ONNX)
# ─────────────────────────────────────────────

PHONE_DETECTION_ENABLED = True

# Ultralytics loads an NCNN export by pointing at the exported model folder
# (the one containing model.ncnn.param / model.ncnn.bin), not a single file.
PHONE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "phone_detection_v2_ncnn_model",
)
PHONE_CLASS_NAME = "phone"
PHONE_IMG_SIZE = 640
PHONE_CONF_THRESHOLD = 0.5
PHONE_CONFIRM_FRAMES = 3
PHONE_CONFIRM_WINDOW = 5
PHONE_COOLDOWN_SEC = 5.0

# YOLO on CPU is heavy; run it every Nth frame to keep the capture/display
# loop from lagging. A held-up phone stays in view long enough to still
# fill the confirmation buffer.
PHONE_DETECT_EVERY_N_FRAMES = 3

# Low-confidence tier for occluded/edge-on/calling-position views, needs a
# longer sustained window since a single low-confidence frame is unreliable.
PHONE_LOW_CONF_THRESHOLD = 0.30
PHONE_LOW_CONF_WINDOW = 10
PHONE_LOW_CONF_FRAMES = 7

# Shape gate: box must look roughly phone-proportioned (~2:1, portrait or
# landscape), filters out other dark handheld objects.
PHONE_MIN_ASPECT = 1.5
PHONE_MAX_ASPECT = 3.0

# Black border added before inference so edge-clipped phones aren't
# penalized for looking incomplete.
PHONE_EDGE_PAD_PX = 60

# ─────────────────────────────────────────────
# Display defaults
# ─────────────────────────────────────────────

DISPLAY_ON_START = True

# ─────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# ─────────────────────────────────────────────
# ffplay output window
# ─────────────────────────────────────────────

DISPLAY_W = 640
DISPLAY_H = 480

# ─────────────────────────────────────────────
# MediaPipe eye landmark indices
# Order per eye: [outer_corner, top_1, top_2, inner_corner, bottom_1, bottom_2]
# ─────────────────────────────────────────────

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# ─────────────────────────────────────────────
# Face landmarker model
# ─────────────────────────────────────────────

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_landmarker.task",
)

MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ─────────────────────────────────────────────
# Alert / frame logging (shared by phone, drowsiness, yawn)
# ─────────────────────────────────────────────

SITE_ID = "demosite-01"
CAMERA_ID = "cam-001"

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)