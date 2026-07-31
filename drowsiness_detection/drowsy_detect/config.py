"""
config.py
---------
All tunable constants for the drowsiness detector live here so the
rest of the codebase never hardcodes a magic number.
"""

import os

# ─────────────────────────────────────────────
# CPU / threading
# ─────────────────────────────────────────────

# Force CPU-only execution (no CUDA) and cap OpenCV/OMP thread usage.
CUDA_VISIBLE_DEVICES = "-1"
OMP_NUM_THREADS = "4"
OPENCV_NUM_THREADS = 4
OPENCV_USE_OPENCL = False


def apply_cpu_settings() -> None:
    """Apply the CPU/threading env vars. Call this before importing cv2/mediapipe."""
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS


# ─────────────────────────────────────────────
# Drowsiness detection thresholds
# ─────────────────────────────────────────────

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
ALERT_COOLDOWN_SEC = 4.0

# ─────────────────────────────────────────────
# Yawn detection thresholds (Mouth Aspect Ratio)
# ─────────────────────────────────────────────

MAR_THRESHOLD = 0.6          # mouth-open ratio above which counts as "open"
YAWN_CONSEC_FRAMES = 20      # ~0.6-0.7s at 30fps held open before it's a yawn
YAWN_COOLDOWN_SEC = 4.0

# MediaPipe face mesh mouth landmark indices:
# top inner lip, bottom inner lip, left corner, right corner
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

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
# Model
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
