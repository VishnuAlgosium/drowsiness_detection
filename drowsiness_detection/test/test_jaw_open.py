"""
test_jaw_open.py
-----------------
Standalone script to test yawn detection using geometric Mouth Aspect
Ratio (MAR) only -- no blendshapes, no jawOpen. MAR is computed purely
from landmark distances (inner-lip gap / mouth width), so it doesn't
depend on skin texture, shading, or appearance -- unlike the jawOpen
blendshape, which should make this more robust across lighting/skin
tone, and should transfer to IR footage without retraining.

A yawn is confirmed only when MAR stays above threshold continuously
for `--yawn-hold` seconds, which filters out brief mouth-open gestures
(talking, a quick tongue-out) that a raw MAR-crossing check alone can't
tell apart from a real yawn.

Usage:
    python -m drowsiness_detection.test.test_jaw_open [--threshold 0.6] [--yawn-hold 1.2]

If you don't have the model file yet:
    wget -O drowsiness_detection/models/face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import argparse
import os
import sys
import time
from collections import deque

# Allow running this file directly (python drowsiness_detection/test/test_jaw_open.py)
# by putting the project root -- two levels up from this file -- on sys.path.
# Without this, Python only adds this file's own directory to sys.path, so
# `import drowsiness_detection...` fails with ModuleNotFoundError even though
# the package is right there one level up.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from drowsiness_detection.drowsy_detect import config
from drowsiness_detection.drowsy_detect.display import LazyDisplay
from drowsiness_detection.drowsy_detect.keyboard_input import KeyReader

import math


# Inner lips -- the vertical gap between these two points is what MAR
# measures growing as the jaw drops.
INNER_LIPS_TOP = 13
INNER_LIPS_BOTTOM = 14

# Outer lip corners -- MAR's horizontal reference (mouth width), so the
# vertical gap is judged relative to face/mouth scale, not raw pixels.
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291


# Inner lip contour (traces the mouth opening itself)
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


# Quad formed by two upper-inner and two lower-inner lip points,
# used to track how the mouth opening's cross-section changes shape.
QUAD_LOOP = [87, 82, 312, 317]  # order matters: this is the loop 87->82->312->317->87





def compute_corner_lift_angles(landmarks, w, h):
    """Angle (degrees) of each mouth corner relative to a horizontal line
    drawn through the mouth's vertical center (midpoint of inner lip
    top/bottom, 13 & 14).

    ~0 deg  -> corner level with center (yawn-like, jaw drops straight down)
    positive -> corner lifted ABOVE center (smile-like, zygomaticus pulls up)
    negative -> corner pulled BELOW center (frown-like)

    Uses atan2 so the sign is meaningful and the angle is well-defined even
    when the corner is nearly level (unlike acos-based angles, which get
    numerically unstable near 0).
    """
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[61]
    right = landmarks[291]

    cx = (top.x + bottom.x) / 2.0 * w
    cy = (top.y + bottom.y) / 2.0 * h

    lx, ly = left.x * w, left.y * h
    rx, ry = right.x * w, right.y * h

    # Image y grows downward, so negate dy to make "up" positive, matching
    # normal angle convention (lift = positive, droop = negative).
    left_angle = math.degrees(math.atan2(-(ly - cy), cx - lx))    # note: left is to the LEFT of center, so dx = cx-lx (positive)
    right_angle = math.degrees(math.atan2(-(ry - cy), rx - cx))   # right corner: dx = rx-cx (positive)

    return left_angle, right_angle, (cx, cy)


def draw_corner_lift_angles(frame, landmarks, w, h, color=(0, 200, 255),
                             base_color=(255, 255, 255)):
    left_angle, right_angle, (cx, cy) = compute_corner_lift_angles(landmarks, w, h)

    cx_i, cy_i = int(cx), int(cy)
    left_pt = (int(landmarks[61].x * w), int(landmarks[61].y * h))
    right_pt = (int(landmarks[291].x * w), int(landmarks[291].y * h))

    # Horizontal reference line through mouth center.
    cv2.line(frame, (0, cy_i), (w, cy_i), base_color, 1)

    # Center-to-corner lines.
    cv2.line(frame, (cx_i, cy_i), left_pt, color, 1)
    cv2.line(frame, (cx_i, cy_i), right_pt, color, 1)
    cv2.circle(frame, (cx_i, cy_i), 3, color, -1)

    cv2.putText(frame, f"{left_angle:+.0f}", (left_pt[0] - 20, left_pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(frame, f"{right_angle:+.0f}", (right_pt[0] + 4, right_pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return left_angle, right_angle



class MonotonicTimestamp:
    """Strictly increasing ms timestamps required by MediaPipe VIDEO mode."""

    def __init__(self):
        self._last_ms = -1

    def next(self) -> int:
        ms = int(time.monotonic() * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        return ms



def draw_landmark_indices(frame, landmarks, indices, w, h, color=(0, 255, 0)):
    """Draws a small dot + the numeric index at each given landmark point.
    Useful for visually mapping index numbers to physical points on the lips."""
    for idx in indices:
        pt = landmarks[idx]
        px, py = int(pt.x * w), int(pt.y * h)
        cv2.circle(frame, (px, py), 2, color, -1)
        cv2.putText(frame, str(idx), (px + 3, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

def draw_all_inner_lip_points(frame, landmarks, w, h):
    for idx in UPPER_LIP_INNER:
        pt = landmarks[idx]
        cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (255, 0, 0), -1)  # blue = upper
    for idx in LOWER_LIP_INNER:
        pt = landmarks[idx]
        cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 255), -1)  # yellow = lower

def build_face_landmarker(model_path: str):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        # No blendshapes needed -- MAR is computed from raw landmark
        # positions only, so this is skipped to save compute per frame.
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def compute_mar(landmarks, w, h) -> float:
    """Geometric Mouth Aspect Ratio: vertical inner-lip gap / mouth width.
    Purely landmark-distance based -- no reliance on skin texture, shading,
    or appearance cues -- so this should transfer to IR imagery without
    retraining or fine-tuning, unlike an appearance-based blendshape."""
    top = landmarks[INNER_LIPS_TOP]
    bottom = landmarks[INNER_LIPS_BOTTOM]
    left = landmarks[MOUTH_LEFT_CORNER]
    right = landmarks[MOUTH_RIGHT_CORNER]

    vertical = (((top.x - bottom.x) * w) ** 2 + ((top.y - bottom.y) * h) ** 2) ** 0.5
    horizontal = (((left.x - right.x) * w) ** 2 + ((left.y - right.y) * h) ** 2) ** 0.5
    # print(f"MAR: vertical={vertical:.2f}, horizontal={horizontal:.2f}, ratio={vertical/horizontal:.2f}")

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def draw_mouth_landmarks(frame, landmarks, w, h, mar_value, threshold):
    """Marks the landmarks MAR is computed from: the inner-lip top/bottom
    gap (colored by open/closed state) plus the outer mouth corners used
    as the width reference."""
    is_open = mar_value > threshold
    color = (0, 0, 255) if is_open else (0, 255, 0)

    top = landmarks[INNER_LIPS_TOP]
    bottom = landmarks[INNER_LIPS_BOTTOM]
    left = landmarks[MOUTH_LEFT_CORNER]
    right = landmarks[MOUTH_RIGHT_CORNER]

    top_pt = (int(top.x * w), int(top.y * h))
    bottom_pt = (int(bottom.x * w), int(bottom.y * h))
    left_pt = (int(left.x * w), int(left.y * h))
    right_pt = (int(right.x * w), int(right.y * h))

    # Mouth-width reference line (dim, for scale only).
    cv2.line(frame, left_pt, right_pt, (120, 120, 120), 1)
    cv2.circle(frame, left_pt, 3, (120, 120, 120), -1)
    cv2.circle(frame, right_pt, 3, (120, 120, 120), -1)

    # The actual MAR gap: highlighted, colored by open/closed state.
    cv2.line(frame, top_pt, bottom_pt, color, 2)
    cv2.circle(frame, top_pt, 4, color, -1)
    cv2.circle(frame, bottom_pt, 4, color, -1)


def draw_hud(frame, mar_value, threshold, fps, w, yawn_count, yawn_confirmed,
             held_sec, yawn_hold, lift_angle=None, is_oscillating=False):
    is_open = mar_value > threshold
    status_color = (0, 0, 255) if is_open else (0, 255, 0)
    status_text = "MOUTH OPEN" if is_open else "MOUTH CLOSED"

    cv2.putText(frame, f"MAR: {mar_value:.3f} (thr {threshold:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, status_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    next_y = 105
    if lift_angle is not None:
        shape_hint = "smile-like" if lift_angle > 8.0 else "yawn-like"
        cv2.putText(frame, f"Corner lift: {lift_angle:+.1f} deg ({shape_hint})",
                    (10, next_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
        next_y += 25

    if is_oscillating:
        cv2.putText(frame, "TALKING? (mouth cycling open/closed)",
                    (10, next_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        next_y += 25

    if is_open:
        cv2.putText(frame, f"Held: {held_sec:.1f}s / {yawn_hold:.1f}s",
                    (10, next_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    next_y += 25

    yawn_color = (0, 0, 255) if yawn_confirmed else (0, 255, 0)
    yawn_text = f"YAWN CONFIRMED  (count: {yawn_count})" if yawn_confirmed else f"Yawns: {yawn_count}"
    cv2.putText(frame, yawn_text, (10, next_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yawn_color, 2)
    next_y += 15

    # Simple horizontal bar visualizing MAR against a 0..1 display range.
    bar_x0, bar_y0 = 10, next_y
    bar_w, bar_h = w - 20, 18
    display_max = max(1.0, threshold * 1.5)
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (80, 80, 80), 1)
    fill_w = int(bar_w * max(0.0, min(1.0, mar_value / display_max)))
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + fill_w, bar_y0 + bar_h), status_color, -1)
    thr_x = bar_x0 + int(bar_w * min(1.0, threshold / display_max))
    cv2.line(frame, (thr_x, bar_y0), (thr_x, bar_y0 + bar_h), (255, 255, 255), 2)


def enhance_low_light(frame):
    """CLAHE (adaptive histogram equalization) on the L channel in LAB space.
    Boosts local contrast without blowing out already-bright areas, which
    helps MediaPipe's landmark model in dim ambient light and with darker
    skin tones, both of which reduce the contrast the model needs at the
    jawline/lips. Does not fix detail genuinely crushed to black -- for
    that, camera-level exposure/IR is the real fix."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def compute_corner_symmetry(landmarks, w, h) -> float:
    """Measures where the mouth-corner line sits relative to the vertical
    top/bottom lip gap: 0.0 = corners exactly centered between top and
    bottom lip ('+' shape, typical of a yawn's jaw-drop), negative = corners
    pulled up toward/above the top lip (typical of a smile/laugh, where the
    zygomaticus muscle lifts the corners), positive = corners pulled down
    toward the bottom lip.

    Purely landmark-position based (no appearance/color), so unlike an
    eye-squint blendshape this should also work on IR footage. This is an
    experimental heuristic -- validate it against your own footage with
    --debug before relying on it; corner lift during yawning/smiling
    varies somewhat by person."""
    top_y = landmarks[INNER_LIPS_TOP].y * h
    bottom_y = landmarks[INNER_LIPS_BOTTOM].y * h
    corner_y = ((landmarks[MOUTH_LEFT_CORNER].y + landmarks[MOUTH_RIGHT_CORNER].y)
                / 2.0) * h

    vertical_gap = bottom_y - top_y
    if vertical_gap <= 0:
        return 0.0

    mid_y = (top_y + bottom_y) / 2.0
    return (corner_y - mid_y) / vertical_gap


def _open_camera():
    """Mirrors detector.py's camera-opening logic exactly, so this test
    script sees the same camera behavior (RTSP vs local, V4L2 backend,
    resolution/FPS/buffer settings) as production."""
    rtsp_url = config.RTSP_URL.strip() if config.RTSP_URL else None

    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
    else:
        cap = cv2.VideoCapture(4, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    parser = argparse.ArgumentParser(description="MAR-only yawn detection test")
    parser.add_argument("--model", type=str,
                         default="../models/face_landmarker.task",
                         help="Path to MediaPipe face_landmarker .task model")
    parser.add_argument("--threshold", type=float, default=0.35,
                         help="MAR above this counts as mouth-open (default 0.6; "
                              "tune against your own footage with --debug)")
    parser.add_argument("--yawn-hold", type=float, default=0.5,
                         help="Seconds MAR must stay above threshold continuously "
                              "before it's confirmed as a yawn, not just any brief "
                              "mouth-open gesture (default 1.2s)")
    parser.add_argument("--debug", action="store_true",
                         help="Print MAR to console once a second")
    parser.add_argument("--enhance", action="store_true",
                         help="Apply CLAHE contrast enhancement before detection "
                              "(helps in low light / with darker skin tones)")
    parser.add_argument("--require-symmetric", action="store_true",
                         help="Additionally require the mouth-corner line to sit "
                              "roughly centered between the top/bottom lip gap "
                              "(rejects smiles/laughs, where corners lift toward "
                              "the top lip). Experimental -- tune "
                              "SYMMETRY_MAX_OFFSET against your own footage first.")
    parser.add_argument("--oscillation-window-sec", type=float, default=3.0,
                     help="Seconds of mouth-open/closed history to scan for "
                          "rapid open/close cycling, i.e. talking (default 3.0s)")
    parser.add_argument("--max-transitions", type=int, default=2,
                         help="If more than this many mouth-open rising edges occur "
                              "within --oscillation-window frames, treat it as "
                              "talking/oscillating and reject the yawn (default 3)")
    
    parser.add_argument("--corner-lift-max", type=float, default=8.0,
                     help="Max avg corner-lift angle (degrees) allowed for a "
                          "yawn; above this the mouth shape looks smile-like "
                          "and is rejected (default 8.0, tune with --debug)")
    args = parser.parse_args()

    display = LazyDisplay(config.DISPLAY_W, config.DISPLAY_H, config.CAM_FPS)
    display.start()
    keys = KeyReader()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("Download it with:")
        print(f"wget -O {args.model} "
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
        sys.exit(1)

    face_mesh = build_face_landmarker(args.model)
    timestamps = MonotonicTimestamp()

    if config.RTSP_URL.strip():
        print(f"[INFO] Using RTSP stream: {config.RTSP_URL.strip()}")
    else:
        print(f"[INFO] Using local webcam (index {config.CAMERA_INDEX})")

    cap = _open_camera()

    if not cap.isOpened():
        print("[ERROR] Cannot open camera/stream")
        sys.exit(1)

    threshold = args.threshold

    print("[INFO] Running. Press 'q' to quit, '+'/'-' to adjust threshold.")

    last_debug_print = 0.0
    smoothed_mar = None
    smoothed_mar = None
    smoothed_lift_angle = None
    SMOOTHING_ALPHA = 0.4  # lower = smoother/slower to react, higher = snappier/noisier

    mouth_open_start = None  # wall-clock time MAR first crossed `threshold`
    yawn_count = 0
    last_yawn_confirmed = False
    mar_during_hold = []  # reset to [] whenever mouth_open_start is reset to None

    YAWN_COOLDOWN_SEC = 4.0
    last_yawn_time = -999.0
    
    mouth_open_history = deque()

    try:
        while True:
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            detect_frame = enhance_low_light(frame) if args.enhance else frame
            rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = face_mesh.detect_for_video(mp_image, timestamps.next())

            mar_value = 0.0
            symmetry = 0.0
            yawn_confirmed = False
            held_sec = 0.0
            is_oscillating = False
            now = time.time()
            SYMMETRY_MAX_OFFSET = 0.25  # tune against your own footage

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                mar_value = compute_mar(landmarks, w, h)
                symmetry = compute_corner_symmetry(landmarks, w, h)

                smoothed_mar = mar_value if smoothed_mar is None else (
                    SMOOTHING_ALPHA * mar_value + (1 - SMOOTHING_ALPHA) * smoothed_mar
                )
               
                
                left_angle, right_angle = draw_corner_lift_angles(frame, landmarks, w, h)
                avg_lift_angle = (left_angle + right_angle) / 2.0

                smoothed_lift_angle = avg_lift_angle if smoothed_lift_angle is None else (
                    SMOOTHING_ALPHA * avg_lift_angle + (1 - SMOOTHING_ALPHA) * smoothed_lift_angle
                )

                if args.debug and (now - last_debug_print > 1.0):
                    print(f"[DEBUG] Corner lift angle -> raw_avg:{avg_lift_angle:+.1f}  smoothed:{smoothed_lift_angle:+.1f}")
               
                is_open_now = smoothed_mar > threshold
                is_open_raw = mar_value > threshold
                mouth_open_history.append((now, is_open_raw))

                mouth_open_history.append((now, is_open_raw))
                while mouth_open_history and now - mouth_open_history[0][0] > args.oscillation_window_sec:
                    mouth_open_history.popleft()
                rising_edges = sum(
                    1 for i in range(1, len(mouth_open_history))
                    if mouth_open_history[i][1] and not mouth_open_history[i-1][1]
                )
                is_oscillating = rising_edges > args.max_transitions

                if args.debug and (now - last_debug_print > 1.0):
                    print(f"[DEBUG] MAR={mar_value:.3f}  corner_symmetry={symmetry:+.3f}  "
                          f"rising_edges={rising_edges}/{args.max_transitions}  "
                          f"oscillating={is_oscillating}")
                    last_debug_print = now
                    
                


                if is_open_now:
                    if mouth_open_start is None:
                        mouth_open_start = now
                        mar_during_hold = []       # reset only when a NEW hold period starts
                    held_sec = now - mouth_open_start
                    duration_ok = held_sec >= args.yawn_hold
                    # A smile/laugh lifts the corners toward (or above) the
                    # top lip, pushing symmetry sharply negative -- reject
                    # those even if duration alone would pass.
                    symmetry_ok = (not args.require_symmetric) or (
                        smoothed_lift_angle < args.corner_lift_max
                    )
                    # inside is_open_now block:
                    mar_during_hold.append(mar_value)
                    mar_variance_ok = (max(mar_during_hold) - min(mar_during_hold)) < 0.5  # tune this

                                        
                    yawn_confirmed = duration_ok and symmetry_ok and mar_variance_ok and not is_oscillating
                    if is_oscillating:
                        # Talking-like cycling detected -- restart the hold
                        # timer so a lucky long "open" phase right after a
                        # burst of chatter can't slip through immediately.
                        mouth_open_start = now
                else:
                    mouth_open_start = None
                    mar_during_hold = []   
            else:
                cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                smoothed_mar = None
                smoothed_lift_angle = None
                mouth_open_start = None
                mar_during_hold = []           # add this too
                mouth_open_history.clear()

            # Count on rising edge only (don't recount every frame the
            # hold condition stays true).
            if yawn_confirmed and not last_yawn_confirmed:
                if now - last_yawn_time >= YAWN_COOLDOWN_SEC:
                    yawn_count += 1
                    last_yawn_time = now
                    print(f"[YAWN #{yawn_count}] confirmed ...")
                else:
                    print(f"[INFO] Yawn-like event ignored (within {YAWN_COOLDOWN_SEC:.0f}s cooldown of last yawn)")
            last_yawn_confirmed = yawn_confirmed

            fps = 1.0 / (time.perf_counter() - frame_start + 1e-6)
            hud_mar = smoothed_mar if smoothed_mar is not None else mar_value
            draw_hud(frame, hud_mar, threshold, fps, w, yawn_count, yawn_confirmed,
                held_sec, args.yawn_hold, lift_angle=smoothed_lift_angle,
                is_oscillating=is_oscillating)

            # LazyDisplay's ffplay subprocess was started with a fixed
            # -video_size (DISPLAY_W x DISPLAY_H). Every frame piped to it
            # must match that exact size in bytes, or ffplay silently fails.
            display_frame = cv2.resize(frame, (config.DISPLAY_W, config.DISPLAY_H))
            display.show(display_frame)

            key = keys.get_key()
            if key:
                key = key.lower()
                if key == "q":
                    print("[INFO] Quit")
                    break
                elif key in ("+", "="):
                    threshold = round(min(2.0, threshold + 0.05), 2)
                    print(f"[INFO] Threshold -> {threshold:.2f}")
                elif key in ("-", "_"):
                    threshold = round(max(0.0, threshold - 0.05), 2)
                    print(f"[INFO] Threshold -> {threshold:.2f}")

    finally:
        keys.restore()
        cap.release()
        display.stop()
        face_mesh.close()

    print(f"[INFO] Session ended. Yawns confirmed: {yawn_count}")


if __name__ == "__main__":
    main()