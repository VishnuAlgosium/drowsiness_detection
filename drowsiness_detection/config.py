"""All tunable constants and static config for the drowsiness detector."""

# ── Detection thresholds ─────────────────────────────────────────────────────
EAR_THRESHOLD      = 0.25   # EAR below this → eye considered closed
CONSEC_FRAMES      = 48     # Consecutive closed-eye frames before alert
ALERT_COOLDOWN_SEC = 4.0    # Min seconds between repeated audio alerts

# ── MediaPipe Face Mesh landmark indices (6-point eye model) ────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Iris indices (drawing only)
LEFT_IRIS_IDX  = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]

# ── Colors (BGR) ─────────────────────────────────────────────────────────────
GREEN      = (29,  158, 117)
RED        = (74,   75, 226)
ORANGE     = (27,  159, 239)
BLUE       = (221, 138,  55)
WHITE      = (255, 255, 255)
BLACK      = (  0,   0,   0)
DARK_PANEL = ( 18,  18,  28)
DARK_BG    = ( 12,  12,  20)

# ── Camera / display ────────────────────────────────────────────────────────
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30
DISPLAY_W, DISPLAY_H = 880, 520

SAVE_DIR = "drowsiness_images"
