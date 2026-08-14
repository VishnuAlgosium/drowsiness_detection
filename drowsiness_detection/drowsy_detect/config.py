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

# RTSP_URL = "rtsp://admin:diffuse123@192.168.0.119:554/cam/realmonitor?channel=1&subtype=1"
RTSP_URL = ""

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
# Distraction detection (head yaw + low-confidence phone)
# ─────────────────────────────────────────────

# MediaPipe face mesh indices used for yaw estimation.
NOSE_TIP_IDX = 1
FACE_LEFT_EDGE_IDX = 234
FACE_RIGHT_EDGE_IDX = 454

# Forward-facing yaw ratio is ~0.5; outside this band counts as looking away.
YAW_RATIO_LOW = 0.35
YAW_RATIO_HIGH = 0.65

DISTRACTION_CONSEC_FRAMES = 45   # ~1.5s at 30fps held before flagging
DISTRACTION_COOLDOWN_SEC = 4.0

# ─────────────────────────────────────────────
# Head drop detection (sudden downward head pitch, e.g. nodding off)
# ─────────────────────────────────────────────

FOREHEAD_IDX = 10
CHIN_IDX = 152

# A real head drop happens quickly; slowly leaning down to check a phone
# should not count. Window over which the "how fast" check is measured.
HEAD_DROP_WINDOW_SEC = 1.0
HEAD_DROP_DELTA = 0.12          # min pitch-ratio rise within the window to count as "sudden"

# Absolute "head is down" threshold, and how long it must stay past that
# threshold to count as a real drop rather than a quick self-correcting nod.
PITCH_RATIO_DOWN = 0.62
HEAD_DROP_HOLD_FRAMES = 10
HEAD_DROP_COOLDOWN_SEC = 4.0

# ─────────────────────────────────────────────
# Phone-use detection (YOLO ONNX)
# ─────────────────────────────────────────────

PHONE_DETECTION_ENABLED = True

# Ultralytics loads an NCNN export by pointing at the exported model folder
# (the one containing model.ncnn.param / model.ncnn.bin), not a single file.
PHONE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "phone_detection_v4_ncnn_model",
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

# ─────────────────────────────────────────────
# Display defaults
# ─────────────────────────────────────────────

# Off by default: most edge deployments (headless Pi) have no attached
# display, and ffplay may not even be installed. Enable with 'v' at runtime.
DISPLAY_ON_START = True

# ─────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# Device index for cv2.VideoCapture when RTSP_URL is unset. On a Pi with a
# single USB/CSI camera this is almost always 0.
CAMERA_INDEX = 2

CAMERA_RECONNECT_ATTEMPTS = 5
CAMERA_RECONNECT_DELAY_SEC = 2.0

# Hard cap so a missing/dark camera can't hang startup forever.
EAR_CALIBRATION_TIMEOUT_SEC = 15.0

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

# Flush/fsync the per-frame CSV every N rows instead of every row, since
# writing at 30fps to an SD card wears it out fast.
FRAME_LOG_FLUSH_EVERY_N = 30

# Old log files older than this are deleted at startup -- SD-card storage
# on edge devices is limited and this runs unattended for long stretches.
LOG_RETENTION_DAYS = 14