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
# Read from environment variable so Docker / docker-compose can inject it.
# Falls back to empty string → USB webcam (CAMERA_INDEX) when not set.
RTSP_URL = os.environ.get("RTSP_URL", "")

# ─────────────────────────────────────────────
# Drowsiness detection (Eye Aspect Ratio)
# ─────────────────────────────────────────────

EAR_THRESHOLD = 0.25
EYES_CLOSED_HOLD_SEC = 0.67   # sustained closure before a drowsiness alert
ALERT_COOLDOWN_SEC = 4.0

# Startup calibration replaces EAR_THRESHOLD with baseline_ear * EAR_THRESHOLD_RATIO,
# since eye shape varies per person. 0 = skip calibration.
EAR_CALIBRATION_FRAMES = 60
EAR_THRESHOLD_RATIO = 0.75

# Calibrated thresholds outside this range are clamped. Guards against a
# squinting driver, or sunglasses at startup, producing a threshold so low
# that drowsiness can never trigger (or so high that every glance does).
EAR_THRESHOLD_MIN = 0.15
EAR_THRESHOLD_MAX = 0.28

# Time-based EMA: alpha = 1 - exp(-dt / tau), so smoothing lag is the same
# in real seconds at 30 fps or 8 fps. 0.065 s matches the old fixed
# alpha=0.4 at 30 fps.
EAR_SMOOTHING_TAU_SEC = 0.065

# Max |left-right| EAR difference, as a fraction of the driver's baseline EAR,
# to count as "both closed" (rejects winks). 0.40 ~= the old absolute 0.12
# for a typical 0.30 baseline, but scales with narrow/wide eyes.
EAR_ASYMMETRY_RATIO = 0.40

# EAR is only trusted when the head is near the calibrated forward pose.
# Looking down at the dashboard narrows the apparent eye opening (false
# "closed"); partial turns foreshorten one eye (real closures rejected as
# winks). Outside these limits the closure timer and PERCLOS are held, not
# updated -- sustained head-down is head-drop's job.
EAR_POSE_YAW_MAX = 25.0
EAR_POSE_PITCH_MAX = 15.0

# Cap on per-frame dt fed to time accumulators, so a capture/inference stall
# followed by one closed-eye frame can't jump a hold timer past its limit.
MAX_FRAME_DT_SEC = 0.2

# Face gone this long is treated as a possible driver change: recalibrate
# EAR/pose baselines in the background when a face returns. 0 = never.
RECALIBRATE_AFTER_FACE_LOSS_SEC = 60.0

# If the face disappears while the driver looked drowsy (eyes closing or head
# down), escalate after this long instead of silently decaying counters --
# a slumped driver can drop out of frame.
FACE_MISSING_ALERT_SEC = 3.0
FACE_MISSING_ALERT_MAX_REPEATS = 3

NO_FACE_GRACE_SEC = 0.17   # face-loss time tolerated before counters start decaying

# ─────────────────────────────────────────────
# Yawn detection (Mouth Aspect Ratio)
# ─────────────────────────────────────────────

MAR_THRESHOLD = 0.35          # mouth-open ratio above which counts as "open"
MAR_SMOOTHING_ALPHA = 0.4    # EMA factor; damps landmark jitter around the threshold
YAWN_HOLD_SEC = 0.5          # seconds mouth must stay open continuously to count as a yawn
YAWN_COOLDOWN_SEC = 4.0
YAWN_MAX_MAR_VARIANCE = 0.5   # max MAR swing during the hold to still count as one steady yawn

# Oscillation filter: a yawn is one open->hold->close; talking/laughing/singing
# opens and closes repeatedly. Reject if rising edges exceed this in the window.
YAWN_MAX_TRANSITIONS = 2

CORNER_LIFT_MAX = 8.0   # max corner-lift angle (degrees) to count as a yawn (rejects smiles)
OSCILLATION_WINDOW_SEC = 3.0   # seconds over which to count rising edges for oscillation

# Smile rejection via mouth-corner lift (see CORNER_LIFT_MAX). Enabled, but
# experimental -- validate against real footage.
YAWN_REQUIRE_SYMMETRIC = True

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
HEAD_DROP_HOLD_SEC = 0.13
HEAD_DROP_COOLDOWN_SEC = 1.0

# ─────────────────────────────────────────────
# Phone-use detection (YOLO NCNN)
# ─────────────────────────────────────────────

PHONE_DETECTION_ENABLED = True

# Ultralytics loads an NCNN export by pointing at the exported model folder
# (containing model.ncnn.param / model.ncnn.bin), not a single file.
PHONE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "phone_detection_v6_ncnn_model",
)
PHONE_CLASS_NAME = "phone"
PHONE_IMG_SIZE = 640
PHONE_CONF_THRESHOLD = 0.7
PHONE_CONFIRM_WINDOW_SEC = 1.0
PHONE_CONFIRM_RATIO = 0.6   # fraction of inferences in the window that must detect
PHONE_COOLDOWN_SEC = 5.0

# YOLO on CPU is heavy; run every Nth frame to keep capture/display from lagging.
PHONE_DETECT_EVERY_N_FRAMES = 3
# Offsets stagger phone/cigarette/seatbelt inference across different frames
# instead of all three landing on the same frame, which smooths per-frame CPU load.
PHONE_DETECT_OFFSET = 0

# Minimum inferences in a window before any YOLO detector can confirm, so a
# single hit can't decide at low FPS. Each window must fit this many inferences.
DETECTOR_MIN_VOTES = 3

# Low-confidence tier for occluded/edge-on/calling-position views; needs a
# longer sustained window since one low-confidence frame is unreliable.
PHONE_LOW_CONF_THRESHOLD = 0.45
PHONE_LOW_CONF_WINDOW_SEC = 2.0
PHONE_LOW_CONF_RATIO = 0.7

# ─────────────────────────────────────────────
# Cigarette detection (YOLO classification model)
# ─────────────────────────────────────────────

CIGARETTE_DETECTION_ENABLED = True

CIGARETTE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "cigarette_classification_v1_ncnn_model",
)
CIGARETTE_CLASS_NAME = "cigarrete"   # the other class is "nocigarette"
CIGARETTE_IMG_SIZE = 224
CIGARETTE_FACE_PADDING = 0.3   # padding ratio around the face box before classifying
CIGARETTE_CONF_THRESHOLD = 0.7
CIGARETTE_CONFIRM_WINDOW_SEC = 1.0
CIGARETTE_CONFIRM_RATIO = 0.6
CIGARETTE_COOLDOWN_SEC = 5.0

# Classification is cheap relative to detection, but keep the same
# every-Nth-frame pattern as phone/seatbelt for a consistent CPU budget.
CIGARETTE_DETECT_EVERY_N_FRAMES = 3
CIGARETTE_DETECT_OFFSET = 1

# ─────────────────────────────────────────────
# Seatbelt detection (YOLO detection model)
# ─────────────────────────────────────────────

SEATBELT_DETECTION_ENABLED = True

SEATBELT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "seatbelt_detection_v1_ncnn_model",
)
SEATBELT_CLASS_NAME = "seatbelt"
SEATBELT_IMG_SIZE = 640
SEATBELT_CONF_THRESHOLD = 0.6
SEATBELT_DETECT_EVERY_N_FRAMES = 3
SEATBELT_DETECT_OFFSET = 2

# The model detects the seatbelt itself, so the violation is its ABSENCE
# over a sustained window, not a detection -- inverse of phone/cigarette.
SEATBELT_ABSENT_WINDOW_SEC = 3.0
SEATBELT_ABSENT_RATIO = 0.8
SEATBELT_COOLDOWN_SEC = 15.0

# Grace period after startup so the driver has time to buckle up before
# absence starts counting toward an alert.
SEATBELT_STARTUP_GRACE_SEC = 8.0

# ─────────────────────────────────────────────
# Display defaults
# ─────────────────────────────────────────────

DISPLAY_ON_START = False   # headless by default on edge devices; toggle with 'v'

# ─────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

CAMERA_INDEX = 4# cv2.VideoCapture device index, used when RTSP_URL is unset

CAMERA_RECONNECT_ATTEMPTS = 5
CAMERA_RECONNECT_DELAY_SEC = 2.0

# How long the background FrameGrabber can go without a new frame before
# the main loop treats the camera as dead and starts reconnecting.
CAMERA_STALE_FRAME_TIMEOUT_SEC = 2.0

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
# Alert / frame logging (shared by all detectors)
# ─────────────────────────────────────────────

SITE_ID = "demosite-01"
CAMERA_ID = "cam-001"

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)

FRAME_LOG_FLUSH_EVERY_N = 30   # flush/fsync every N rows instead of every row (SD-card wear)
LOG_RETENTION_DAYS = 14        # delete log files older than this at startup


# ─────────────────────────────────────────────
# Blink visibility detection (IR-blocking sunglasses, etc.)
# ─────────────────────────────────────────────
NO_BLINK_TIMEOUT_SEC = 10.0
MAX_BLINK_SEC = 0.5   # longer closures are sustained closure, not a blink
OCCLUSION_ALERT_REPEAT_SEC = 30.0
OCCLUSION_HEAD_DROP_HOLD_SEC = 0.07

# ─────────────────────────────────────────────
# PERCLOS (rolling percentage of eye closure)
# ─────────────────────────────────────────────
PERCLOS_WINDOW_SEC = 60.0       
PERCLOS_ALERT_THRESHOLD = 0.40  
PERCLOS_COOLDOWN_SEC = 10.0
PERCLOS_MIN_COVERAGE_SEC = 30.0
PERCLOS_RESET_AFTER_FACE_LOSS_SEC = 10.0

# ────────────────────────────────────────────
# Camera-block detection (lens occlusion, IR-blocking sunglasses, etc.)
# ─────────────────────────────────────────────
CAMERA_BLOCK_DARK_MEAN=25.5
CAMERA_BLOCK_MIN_STD=12.0
CAMERA_BLOCK_MIN_LAPLACIAN_VAR=15.0
CAMERA_PARTIAL_BLOCK_RATIO=0.25 
CAMERA_PARTIAL_LOST_EDGE_FRAC=0.2
CAMERA_REFERENCE_ALPHA=0.02
CAMERA_BLOCK_HOLD_SEC=2.0
CAMERA_BLOCK_CLEAR_SEC=1.0
CAMERA_BLOCK_CHECK_EVERY_N_FRAMES=3


# ─────────────────────────────────────────────
# Driver recognition (face_identifier.py + driver_identity.py)
# ─────────────────────────────────────────────
DRIVER_ID_ENABLED = True
DRIVER_ID_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "mobilefacenet_int8.tflite"
)
DRIVER_ID_EMPLOYEE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "employee"
)
DRIVER_ID_THRESHOLD = 0.50              # cosine similarity for a KNOWN match -- tune on your photos
DRIVER_ID_VOTES = 5                     # frames sampled to decide who is driving
DRIVER_ID_MIN_AGREEMENT = 0.6           # fraction of those that must agree (3 of 5)
DRIVER_ID_SAMPLE_EVERY_SEC = 0.2        # sampling rate while deciding
DRIVER_ID_REVERIFY_SEC = 30.0           # spot-check interval once decided
DRIVER_ID_REVERIFY_MISMATCHES = 3       # consecutive disagreeing spot-checks before re-identifying
DRIVER_ID_RESET_AFTER_FACE_LOSS_SEC = 10.0   # face gone this long -> identify again
DRIVER_ID_ALERT_UNKNOWN = True          # sound + log when the driver isn't enrolled
DRIVER_ID_UNKNOWN_REPEAT_SEC = 60.0     # repeat the unknown-driver alert while it persists
DRIVER_PROFILE_PATH = os.path.join(LOG_DIR, "driver_profiles.json")   # per-driver EAR baselines