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
    """Apply CPU/threading env vars. Call before importing cv2/mediapipe/ultralytics."""
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS

# RTSP_URL = "rtsp://admin:diffuse123@192.168.0.119:554/cam/realmonitor?channel=1&subtype=1"
RTSP_URL = ""

# ─────────────────────────────────────────────
# Drowsiness detection (Eye Aspect Ratio)
# ─────────────────────────────────────────────

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
ALERT_COOLDOWN_SEC = 4.0

# Startup calibration replaces EAR_THRESHOLD with baseline_ear * EAR_THRESHOLD_RATIO,
# since eye shape varies per person. 0 = skip calibration.
EAR_CALIBRATION_FRAMES = 60
EAR_THRESHOLD_RATIO = 0.75

EAR_SMOOTHING_ALPHA = 0.4   # EMA factor; lower = smoother, damps blink/occlusion noise
EAR_ASYMMETRY_MAX = 0.12   # max |left-right| EAR diff to count as "both closed" (rejects winks)

NO_FACE_GRACE_FRAMES = 5   # no-face frames tolerated before counters start decaying

# ─────────────────────────────────────────────
# Yawn detection (Mouth Aspect Ratio)
# ─────────────────────────────────────────────

MAR_THRESHOLD = 0.3          # mouth-open ratio above which counts as "open"
MAR_SMOOTHING_ALPHA = 0.4    # EMA factor; damps landmark jitter around the threshold
YAWN_HOLD_SEC = 1.0          # seconds mouth must stay open continuously to count as a yawn
YAWN_COOLDOWN_SEC = 0.5

# Oscillation filter: a yawn is one open->hold->close; talking/laughing/singing
# opens and closes repeatedly. Reject if rising edges exceed this in the window.
YAWN_TRANSITION_WINDOW = 30
YAWN_MAX_TRANSITIONS = 1

# Optional smile/laugh rejection, off by default -- experimental, validate
# against real footage before enabling.
YAWN_REQUIRE_SYMMETRIC = True
YAWN_SYMMETRY_MAX_OFFSET = 0.25

# MediaPipe mouth landmarks: top inner lip, bottom inner lip, left corner, right corner
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# Outer corners, used only for the symmetry check above.
MOUTH_OUTER_LEFT = 61
MOUTH_OUTER_RIGHT = 291

# ─────────────────────────────────────────────
# Distraction detection (head pose + low-confidence phone)
# ─────────────────────────────────────────────

NOSE_TIP_IDX = 1

# Tolerance around the calibrated baseline (see _calibrate_baseline in
# detector.py), since an off-center mount means "forward" isn't 0 degrees.
YAW_ANGLE_MAX = 20.0
PITCH_ANGLE_MAX = 20.0
ROLL_ANGLE_MAX = 25.0



DISTRACTION_HOLD_SEC = 0.5   # look-away hold time in real seconds (fps-independent)
DISTRACTION_COOLDOWN_SEC = 4.0

# ─────────────────────────────────────────────
# Head drop detection (sudden downward pitch, e.g. nodding off)
# ─────────────────────────────────────────────

FOREHEAD_IDX = 10
CHIN_IDX = 152

# A real head drop is fast; slowly leaning down to check a phone shouldn't
# count. Window over which the "how fast" rise is measured.
HEAD_DROP_WINDOW_SEC = 0.5
HEAD_DROP_DELTA = 0.12   # min pitch-ratio rise within the window to count as "sudden"

HEAD_DROP_SMOOTHING_ALPHA = 0.8   # EMA factor; damps landmark jitter

# "Head is down" threshold, and how long it must hold to count as a real
# drop rather than a quick glance or self-correcting nod.
PITCH_RATIO_DOWN = 0.62
HEAD_DROP_HOLD_FRAMES = 4
HEAD_DROP_COOLDOWN_SEC = 1.0

# ─────────────────────────────────────────────
# Phone-use detection (YOLO ONNX)
# ─────────────────────────────────────────────

PHONE_DETECTION_ENABLED = True

# Ultralytics loads an NCNN export by pointing at the exported model folder
# (containing model.ncnn.param / model.ncnn.bin), not a single file.
PHONE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "phone_detection_v4_ncnn_model",
)
PHONE_CLASS_NAME = "phone"
PHONE_IMG_SIZE = 640
PHONE_CONF_THRESHOLD = 0.7
PHONE_CONFIRM_FRAMES = 3
PHONE_CONFIRM_WINDOW = 5
PHONE_COOLDOWN_SEC = 5.0

# YOLO on CPU is heavy; run every Nth frame to keep capture/display from lagging.
PHONE_DETECT_EVERY_N_FRAMES = 3

# Low-confidence tier for occluded/edge-on/calling-position views; needs a
# longer sustained window since one low-confidence frame is unreliable.
PHONE_LOW_CONF_THRESHOLD = 1.0
PHONE_LOW_CONF_WINDOW = 10
PHONE_LOW_CONF_FRAMES = 7

# ─────────────────────────────────────────────
# Display defaults
# ─────────────────────────────────────────────

DISPLAY_ON_START = True   # off by default on headless edge devices; toggle with 'v'

# ─────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

CAMERA_INDEX = 0   # cv2.VideoCapture device index, used when RTSP_URL is unset

CAMERA_RECONNECT_ATTEMPTS = 5
CAMERA_RECONNECT_DELAY_SEC = 2.0

EAR_CALIBRATION_TIMEOUT_SEC = 15.0   # hard cap so a missing/dark camera can't hang startup

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

FRAME_LOG_FLUSH_EVERY_N = 30   # flush/fsync every N rows instead of every row (SD-card wear)
LOG_RETENTION_DAYS = 14        # delete log files older than this at startup