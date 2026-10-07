"""
detector.py
-----------
Ties together camera capture, MediaPipe FaceLandmarker (EAR/MAR/head pose),
the YOLO detectors (phone, cigarette, seatbelt), display, keyboard control,
and audio/log alerting into one shared run loop.
"""

import math
import os
import signal
import sys
import time
from collections import deque

import cv2

from . import config
from .alerts import log_alert, FrameLogger, cleanup_old_logs, set_alert_context
from .audio import (
    play_alert, play_yawn_alert, play_phone_alert, play_distraction_alert,
    play_head_drop_alert, play_occlusion_alert, play_perclos_alert,
    play_cigarette_alert, play_seatbelt_alert, shutdown_audio, play_unknown_driver_alert,
)
from .driver_identity import DriverIdentityTracker, DriverProfiles, UNKNOWN
from .ear import eye_aspect_ratio, average_ear
from .mar import mouth_aspect_ratio
from .gaze import head_pose_angles, head_pitch_ratio
from .face_crop import padded_face_box
from .blink_visibility import BlinkVisibilityMonitor
from .perclos import PerclosMonitor
from .calibration import Baseline, BaselineCalibrator, default_baseline
from .capture import FrameGrabber
from .phone import PhoneDetector
from .cigarette import CigaretteDetector
from .seatbelt import SeatbeltDetector
from .display import LazyDisplay
from .keyboard_input import KeyReader
from .camera_health import open_camera_with_retry, CameraOcclusionMonitor


_shutdown_requested = False


def _ema_alpha(dt: float, tau_sec: float) -> float:
    """Time-based EMA factor, so smoothing lag is fps-independent."""
    if dt <= 0 or tau_sec <= 0:
        return 1.0 if tau_sec <= 0 else 0.0
    return 1.0 - math.exp(-dt / tau_sec)


def _print_baseline(baseline: Baseline) -> None:
    clamp_note = " (clamped to configured range)" if baseline.clamped else ""
    print(f"[INFO] Baseline EAR={baseline.ear:.3f} -> drowsy threshold={baseline.ear_threshold:.3f}{clamp_note}")
    print(f"[INFO] Baseline gaze yaw={baseline.yaw:.1f} pitch={baseline.pitch:.1f} roll={baseline.roll:.1f}")


def _handle_sigterm(signum, frame) -> None:
    """Sets a flag instead of exiting immediately, so the main loop's
    `finally` block still runs (releases camera, closes audio/log files)."""
    global _shutdown_requested
    _shutdown_requested = True


class MonotonicTimestamp:
    """Strictly increasing ms timestamps for MediaPipe's VIDEO mode, which
    requires each timestamp to exceed the last. Wall-clock time can jump
    backward on NTP sync; time.monotonic() never does."""

    def __init__(self):
        self._last_ms = -1

    def next(self) -> int:
        ms = int(time.monotonic() * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        return ms


def _open_camera():
    rtsp_url = config.RTSP_URL.strip() if config.RTSP_URL else None

    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _check_model() -> None:
    if not os.path.exists(config.MODEL_PATH):
        print(f"[ERROR] Missing {config.MODEL_PATH}")
        print("Download it first:")
        print(f"wget -O {config.MODEL_PATH} {config.MODEL_DOWNLOAD_URL}")
        sys.exit(1)

    ncnn_models = [
        (config.PHONE_DETECTION_ENABLED, config.PHONE_MODEL_PATH, "phone"),
        (config.CIGARETTE_DETECTION_ENABLED, config.CIGARETTE_MODEL_PATH, "cigarette"),
        (config.SEATBELT_DETECTION_ENABLED, config.SEATBELT_MODEL_PATH, "seatbelt"),
    ]
    for enabled, model_path, label in ncnn_models:
        if not enabled:
            continue

        param_file = os.path.join(model_path, "model.ncnn.param")
        bin_file = os.path.join(model_path, "model.ncnn.bin")

        if not (os.path.exists(param_file) and os.path.exists(bin_file)):
            print(f"[ERROR] Missing NCNN model files under {model_path}")
            print(f"Expected model.ncnn.param and model.ncnn.bin in that folder, or set "
                  f"config.{label.upper()}_DETECTION_ENABLED = False to skip {label} detection.")
            sys.exit(1)


def _check_inference_stagger() -> None:
    """Warn if the YOLO detectors can land on the same frame (or never run)."""
    intervals = {
        config.PHONE_DETECT_EVERY_N_FRAMES,
        config.CIGARETTE_DETECT_EVERY_N_FRAMES,
        config.SEATBELT_DETECT_EVERY_N_FRAMES,
    }
    offsets = [config.PHONE_DETECT_OFFSET, config.CIGARETTE_DETECT_OFFSET, config.SEATBELT_DETECT_OFFSET]
    if len(intervals) != 1 or len(set(offsets)) != len(offsets) or max(offsets) >= min(intervals):
        print("[WARN] YOLO *_DETECT_EVERY_N_FRAMES / *_DETECT_OFFSET overlap -- models may share frames")


def _build_driver_identifier():
    """FaceIdentifier with the employee database loaded, or None (recognition off).
    Optional: any problem disables recognition with a warning instead of
    stopping driver monitoring."""
    if not getattr(config, "DRIVER_ID_ENABLED", False):
        return None
    try:
        from .face_identifier import FaceIdentifier
        identifier = FaceIdentifier(
            model_path=config.DRIVER_ID_MODEL_PATH,
            employee_dir=config.DRIVER_ID_EMPLOYEE_DIR,
            threshold=config.DRIVER_ID_THRESHOLD,
        )
        if identifier.load_employee_database() == 0:
            print("[WARN] Driver recognition: no enrolled employees -- recognition disabled")
            return None
        return identifier
    except Exception as e:
        print(f"[WARN] Driver recognition disabled: {e}")
        return None


def _build_face_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.FaceLandmarker.create_from_options(options), mp


def _face_problem(landmarks, gray_flipped, w, h, left_ear, right_ear, yaw, pitch):
    """Return None if this face is good enough to calibrate on, else a short
    human-readable reason. Rejects faces that are cut off, too small, have an
    eye hidden/closed, are turned away, or have part of the face covered."""
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    margin = getattr(config, "CALIB_FACE_EDGE_MARGIN", 0.02)
    if min(xs) < margin or max(xs) > 1 - margin or min(ys) < margin or max(ys) > 1 - margin:
        return "Face partly out of frame"

    if max(xs) - min(xs) < getattr(config, "CALIB_MIN_FACE_WIDTH", 0.15):
        return "Face too far from camera"

    if min(left_ear, right_ear) < getattr(config, "CALIB_MIN_OPEN_EAR", 0.15):
        return "Eyes closed or hidden"
    if abs(left_ear - right_ear) > getattr(config, "CALIB_MAX_EAR_ASYMMETRY", 0.10):
        return "One eye hidden / face partly covered"

    max_angle = getattr(config, "CALIB_MAX_HEAD_ANGLE", 30.0)
    if abs(yaw) > max_angle or abs(pitch) > max_angle:
        return "Look straight at the road"

    # Every quarter of the face must be visible (not dark, not a flat blob).
    x1, x2 = int(min(xs) * w), int(max(xs) * w)
    y1, y2 = int(min(ys) * h), int(max(ys) * h)
    face = gray_flipped[max(0, y1):y2, max(0, x1):x2]
    if face.size == 0:
        return "No face detected"
    fh, fw = face.shape[:2]
    dark = getattr(config, "CAMERA_BLOCK_DARK_MEAN", 25.0)
    min_std = getattr(config, "CALIB_FACE_MIN_STD", 8.0)
    for qy in (0, fh // 2):
        for qx in (0, fw // 2):
            q = face[qy:qy + fh // 2, qx:qx + fw // 2]
            if q.size and (q.mean() < dark or q.std() < min_std):
                return "Face partly covered"
    return None


def _face_calibration_sample(frame, face_mesh, mp, timestamps: MonotonicTimestamp):
    """Run the face landmarker on one frame.
    Returns (sample, problem): sample is (ear, yaw, pitch, roll) when the face
    is good, else None with `problem` saying why."""
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    face_results = face_mesh.detect_for_video(mp_image, timestamps.next())

    if not face_results.face_landmarks:
        return None, "No face detected"

    landmarks = face_results.face_landmarks[0]
    left_ear = eye_aspect_ratio(landmarks, config.LEFT_EYE_IDX, w, h)
    right_ear = eye_aspect_ratio(landmarks, config.RIGHT_EYE_IDX, w, h)

    yaw, pitch, roll = 0.0, 0.0, 0.0
    if face_results.facial_transformation_matrixes:
        pose_angles = head_pose_angles(face_results.facial_transformation_matrixes[0])
        if pose_angles:
            yaw, pitch, roll = pose_angles

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    problem = _face_problem(landmarks, gray, w, h, left_ear, right_ear, yaw, pitch)
    if problem:
        return None, problem

    return ((left_ear + right_ear) / 2.0, yaw, pitch, roll), None


def _show_calibration(ui, frame, lines, color):
    """Draw calibration status on the live view (no-op if display is off)."""
    if not ui["on"]:
        return
    view = cv2.flip(frame, 1)
    for i, text in enumerate(lines):
        cv2.putText(view, text, (10, 40 + i * 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 if i == 0 else 0.6, color, 2)
    if (view.shape[1], view.shape[0]) != (config.DISPLAY_W, config.DISPLAY_H):
        view = cv2.resize(view, (config.DISPLAY_W, config.DISPLAY_H))
    ui["display"].show(view)


def _calibrate_baseline(grabber: FrameGrabber, face_mesh, mp, timestamps: MonotonicTimestamp,
                        camera_monitor: CameraOcclusionMonitor, ui, keys):
    """
    Collect config.EAR_CALIBRATION_FRAMES GOOD frames and derive:
      - a personal EAR threshold (eye shape varies per person)
      - a yaw/pitch/roll baseline (off-center camera mount)

    A frame is only used if:
      - the camera is not fully or partly blocked, AND
      - a face is detected that is fully in frame, big enough, roughly
        facing forward, both eyes open and visible, no part of it covered.

    Otherwise calibration WAITS (it does not fall back to defaults), shows
    the reason on screen/console, and plays an alert if the problem lasts
    CALIB_PROBLEM_ALERT_SEC (repeating every CAMERA_BLOCK_ALERT_REPEAT_SEC).
    If good frames stop for CALIB_RESET_AFTER_SEC, it restarts from 0/N so the
    baseline always comes from one clean, continuous look.

    CALIBRATION_MAX_WAIT_SEC > 0 allows a fallback to defaults after that
    long; 0 (default) = wait until calibration succeeds.

    Keys: q = quit, v = toggle display.
    Returns (baseline, pending_calibrator), or None if the user quit.
    baseline comes from calibration.BaselineCalibrator: medians (blinks can't
    drag it down) and the threshold clamped to EAR_THRESHOLD_MIN..MAX.
    pending_calibrator is None, except after a CALIBRATION_MAX_WAIT_SEC
    fallback: then it holds the good samples so far and the main loop keeps
    filling it from good frames, applying the result when it completes.
    """
    target = config.EAR_CALIBRATION_FRAMES
    if target <= 0:
        return default_baseline(), None

    repeat_sec = getattr(config, "CAMERA_BLOCK_ALERT_REPEAT_SEC", 10.0)
    alert_after = getattr(config, "CALIB_PROBLEM_ALERT_SEC", 3.0)
    reset_after = getattr(config, "CALIB_RESET_AFTER_SEC", 1.5)
    max_wait = getattr(config, "CALIBRATION_MAX_WAIT_SEC", 0.0)
    GREEN, ORANGE, RED = (0, 255, 0), (0, 165, 255), (0, 0, 255)

    print(f"[INFO] Calibrating baseline ({target} frames, "
          "keep eyes open and look at the road normally)...")

    samples = []   # good (ear, yaw, pitch, roll) samples of one continuous look
    last_frame_id = -1
    wait_start = time.monotonic()
    collect_start = None
    bad_since = None
    last_problem = None
    last_problem_print = -math.inf
    last_alert = -math.inf
    last_progress_print = 0

    while len(samples) < target:
        if _shutdown_requested:
            return None

        key = keys.get_key()
        if key:
            key = key.lower()
            if key == "q":
                print("[INFO] Quit during calibration")
                return None
            if key == "v":
                ui["on"] = not ui["on"]
                if ui["on"]:
                    ui["display"].start()
                    print("[INFO] Display ON")
                else:
                    ui["display"].stop()
                    print("[INFO] Display OFF")

        frame, frame_id, _ = grabber.read()
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.002)
            continue
        last_frame_id = frame_id
        now = time.monotonic()

        if max_wait and now - wait_start > max_wait:
            print(f"[WARN] Calibration not possible for {max_wait:.0f}s -- using defaults; "
                  "will finish calibrating in the background from good frames")
            log_alert("calibration_failed", {"last_problem": last_problem, "good_samples": len(samples)})
            pending = BaselineCalibrator(target)
            for sample in samples:
                pending.add(*sample)
            return default_baseline(), pending

        blocked = camera_monitor.update(frame, now)
        if blocked or camera_monitor.suspect:
            reason = camera_monitor.reason
            problem = ("Camera partly blocked" if reason.startswith("partially")
                       else "Camera blocked") + f" ({reason})"
        else:
            sample, problem = _face_calibration_sample(frame, face_mesh, mp, timestamps)

        if problem is None:
            if collect_start is None:
                collect_start = now
            samples.append(sample)
            camera_monitor.update_reference(alpha=0.2)   # learn the clear view
            if bad_since is not None:
                print(f"[INFO] Face OK -- calibrating ({len(samples)}/{target})")
            bad_since = None
            last_problem = None
            if len(samples) - last_progress_print >= 10 or len(samples) == target:
                print(f"[INFO] Calibrating... {len(samples)}/{target}")
                last_progress_print = len(samples)
            _show_calibration(ui, frame, [f"Calibrating {len(samples)}/{target}",
                                          "Keep eyes open, look at the road"], GREEN)
            continue

        # ── bad frame: don't use it ──
        if bad_since is None:
            bad_since = now
        if problem != last_problem and now - last_problem_print > 1.0:
            print(f"[WARN] Calibration waiting: {problem}")
            last_problem = problem
            last_problem_print = now

        if samples and now - bad_since > reset_after:
            print(f"[WARN] Calibration interrupted at {len(samples)}/{target} -- restarting from 0")
            samples.clear()
            collect_start = None
            last_progress_print = 0

        if now - bad_since >= alert_after and now - last_alert > repeat_sec:
            print(f"[CALIBRATION ALERT] {problem} -- cannot calibrate")
            play_occlusion_alert()
            log_alert("calibration_blocked", {"problem": problem})
            last_alert = now

        color = RED if problem.startswith("Camera") or problem == "No face detected" else ORANGE
        _show_calibration(ui, frame, [f"Calibrating {len(samples)}/{target} - PAUSED", problem], color)

    calibrator = BaselineCalibrator(len(samples))
    for sample in samples:
        calibrator.add(*sample)
    baseline = calibrator.result()

    took = time.monotonic() - collect_start if collect_start is not None else 0.0
    print(f"[INFO] Calibration complete: {len(samples)}/{target} good face frames in {took:.1f}s")
    _print_baseline(baseline)
    return baseline, None


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
    top = landmarks[config.MOUTH_TOP]
    bottom = landmarks[config.MOUTH_BOTTOM]
    left = landmarks[config.MOUTH_OUTER_LEFT]
    right = landmarks[config.MOUTH_OUTER_RIGHT]

    cx = (top.x + bottom.x) / 2.0 * w
    cy = (top.y + bottom.y) / 2.0 * h

    lx, ly = left.x * w, left.y * h
    rx, ry = right.x * w, right.y * h

    # Image y grows downward, so negate dy to make "up" positive, matching
    # normal angle convention (lift = positive, droop = negative).
    left_angle = math.degrees(math.atan2(-(ly - cy), cx - lx))    # left is LEFT of center, so dx = cx-lx (positive)
    right_angle = math.degrees(math.atan2(-(ry - cy), rx - cx))   # right corner: dx = rx-cx (positive)

    return left_angle, right_angle


def run() -> None:
    config.apply_cpu_settings()

    cv2.ocl.setUseOpenCL(config.OPENCV_USE_OPENCL)
    cv2.setNumThreads(config.OPENCV_NUM_THREADS)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    _check_model()
    _check_inference_stagger()
    cleanup_old_logs()

    face_mesh, mp = _build_face_landmarker()
    phone_detector = PhoneDetector() if config.PHONE_DETECTION_ENABLED else None
    cigarette_detector = CigaretteDetector() if config.CIGARETTE_DETECTION_ENABLED else None
    seatbelt_detector = SeatbeltDetector() if config.SEATBELT_DETECTION_ENABLED else None
    driver_identifier = _build_driver_identifier()

    if config.RTSP_URL.strip():
        print(f"[INFO] Using RTSP stream: {config.RTSP_URL.strip()}")
    else:
        print(f"[INFO] Using local webcam (index {config.CAMERA_INDEX})")

    def _camera_unavailable():
        print("[ALERT] Camera not available -- will keep retrying")
        play_occlusion_alert()
        log_alert("camera_unavailable", {"source": config.RTSP_URL.strip() or config.CAMERA_INDEX})

    cap = open_camera_with_retry(
        _open_camera,
        should_abort=lambda: _shutdown_requested,
        on_unavailable=_camera_unavailable,
    )
    if cap is None:
        print("[ERROR] Cannot open camera, exiting")
        shutdown_audio()
        sys.exit(1)

    grabber = FrameGrabber(cap)

    # One monitor shared by calibration and the main loop, so a block that
    # starts during calibration is still tracked once the loop begins.
    camera_monitor = CameraOcclusionMonitor()
    camera_block_repeat_sec = getattr(config, "CAMERA_BLOCK_ALERT_REPEAT_SEC", 10.0)

    # Start the display and keyboard BEFORE calibration so progress is visible
    # and q / v work while calibrating.
    display = LazyDisplay(config.DISPLAY_W, config.DISPLAY_H, config.CAM_FPS)
    ui = {"display": display, "on": config.DISPLAY_ON_START}
    if ui["on"]:
        display.start()
    keys = KeyReader()

    timestamps = MonotonicTimestamp()
    try:
        calibration = _calibrate_baseline(grabber, face_mesh, mp, timestamps, camera_monitor, ui, keys)
    except BaseException:
        keys.restore(); display.stop(); grabber.release(); cap.release(); shutdown_audio()
        raise
    if calibration is None:
        keys.restore(); display.stop(); grabber.release(); cap.release(); shutdown_audio()
        print("[INFO] Exited before calibration finished")
        return
    baseline, pending_calibrator = calibration
    baseline_ear = baseline.ear
    ear_threshold = baseline.ear_threshold
    gaze_baseline_yaw, gaze_baseline_pitch, gaze_baseline_roll = baseline.yaw, baseline.pitch, baseline.roll
    recalibration_requested = False   # one background recalibration per long face-loss episode
    display_on = ui["on"]

    frame_log = FrameLogger()

    # ── Driver recognition state ──
    identity = DriverIdentityTracker() if driver_identifier is not None else None
    profiles = DriverProfiles()
    unknown_alert_count = 0
    last_unknown_alert = -math.inf
    blocked_since = None
    set_alert_context()

    eyes_closed_sec = 0.0
    alert_count = 0
    last_alert = -math.inf
    smoothed_ear = None
    face_lost_at = None
    eyes_occluded = False
    occlusion_alert_count = 0
    last_occlusion_alert = -math.inf
    blink_monitor = BlinkVisibilityMonitor(
        config.NO_BLINK_TIMEOUT_SEC, ear_threshold, config.MAX_BLINK_SEC
    )

    perclos_monitor = PerclosMonitor(config.PERCLOS_WINDOW_SEC, config.PERCLOS_MIN_COVERAGE_SEC)
    current_perclos = 0.0
    perclos_ready = False

    eyes_closed = False
    ear_pose_reliable = False
    drowsy_at_face_loss = False       # was the driver drowsy-looking when the face disappeared?
    face_missing_alert_repeats = 0
    perclos_alert_count = 0
    last_perclos_alert = -math.inf

    smoothed_mar = None
    mouth_open_start = None   # time MAR first crossed MAR_THRESHOLD, or None
    yawn_count = 0
    last_yawn_confirmed = False   # previous frame's yawn_confirmed, to count on rising edge only
    mouth_open_history = deque()
    mar_during_hold = []

    gaze_away_start = None
    gaze_away_elapsed = 0.0
    gaze_distraction_count = 0
    last_gaze_alert = -math.inf
    current_yaw_deg = 0.0
    current_pitch_deg = 0.0
    current_roll_deg = 0.0

    head_down_sec = 0.0
    head_drop_count = 0
    last_head_drop_alert = -math.inf
    current_pitch_ratio = 0.5
    smoothed_pitch_ratio = None
    is_head_down = False
    pitch_history = deque()  # (timestamp, smoothed_pitch_ratio) pairs, evicted by wall-clock age

    last_yawn_time = -math.inf
    smoothed_lift_angle = None
    yawn_confirmed = False
    held_sec = 0.0

    camera_blocked = camera_monitor.blocked
    camera_block_alert_count = 0
    last_camera_block_alert = -math.inf

    # Driver face missing (camera clear, but nobody / face turned fully away)
    no_face_alert_sec = getattr(config, "NO_FACE_ALERT_SEC", 5.0)
    no_face_repeat_sec = getattr(config, "NO_FACE_ALERT_REPEAT_SEC", 10.0)
    no_face_alert_count = 0
    last_no_face_alert = -math.inf
    face_missing_sec = 0.0

    frame_num = 0
    last_frame_id = -1
    prev_frame_time = None
    session_start_time = time.monotonic()

    print()
    print("[INFO] Running")
    print("[INFO] Keys: v = toggle display, q = quit")

    try:
        while True:
            if _shutdown_requested:
                print("[INFO] Shutdown signal received")
                break

            frame, frame_id, frame_age_sec = grabber.read()

            # ── Camera disconnected / stale -> keep retrying until it's back ──
            if frame_age_sec > config.CAMERA_STALE_FRAME_TIMEOUT_SEC:
                print("[WARN] Camera frame failed, attempting reconnect")
                log_alert("camera_disconnected", {"frame_age_sec": round(frame_age_sec, 1)})
                play_occlusion_alert()
                grabber.release()
                cap.release()

                cap = open_camera_with_retry(_open_camera, should_abort=lambda: _shutdown_requested)
                if cap is None:
                    print("[ERROR] Camera reconnect failed, exiting")
                    cap = cv2.VideoCapture()   # dummy so finally: cap.release() is safe
                    break

                grabber = FrameGrabber(cap)
                camera_monitor.reset(keep_reference=False)
                camera_blocked = False
                last_frame_id = -1
                prev_frame_time = None
                continue

            if frame is None or frame_id == last_frame_id:
                time.sleep(0.002)  # no new frame yet; avoid busy-spinning
                continue
            last_frame_id = frame_id

            frame_start = time.perf_counter()
            now = time.monotonic()
            frame_dt = now - prev_frame_time if prev_frame_time is not None else 0.0
            # Cap so one stall + one closed frame can't jump a hold timer past its limit.
            frame_dt = min(frame_dt, config.MAX_FRAME_DT_SEC)
            prev_frame_time = now

            # ── Camera blocked check (runs every frame, whole session) ──
            was_blocked = camera_blocked
            camera_blocked = camera_monitor.update(frame, now)

            if camera_blocked and not was_blocked:
                print("[WARN] Camera blocked -- detection paused until the view is clear")
            # Alert only while the frame really still looks blocked (not during the
            # short "confirming clear" hold, when the view is already fine).
            if camera_blocked and camera_monitor.suspect and now - last_camera_block_alert > camera_block_repeat_sec:
                camera_block_alert_count += 1
                print(f"[CAMERA BLOCKED #{camera_block_alert_count}] reason={camera_monitor.reason} "
                      f"mean={camera_monitor.mean:.1f} std={camera_monitor.std:.1f} edges={camera_monitor.edges:.1f}")
                play_occlusion_alert()
                log_alert("camera_blocked", {
                    "phase": "running",
                    "reason": camera_monitor.reason,
                    "mean": round(camera_monitor.mean, 1),
                    "std": round(camera_monitor.std, 1),
                    "edges": round(camera_monitor.edges, 1),
                })
                last_camera_block_alert = now
            elif was_blocked and not camera_blocked:
                print("[INFO] Camera view clear again -- detection resumed")
                log_alert("camera_unblocked", {"phase": "running"})
                last_camera_block_alert = -math.inf   # alert immediately if it gets blocked again

            frame_num += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            current_ear = 0.0
            current_mar = 0.0
            face_crop = None
            perclos_ready = False      # only alert on PERCLOS from frames with a face
            ear_pose_reliable = False
            identity_events = []
            face_bbox = None

            # While blocked, skip the face model entirely: a covered lens has no
            # face, and running on it just wastes CPU.
            face_found = False
            if not camera_blocked:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                face_results = face_mesh.detect_for_video(mp_image, timestamps.next())
                face_found = bool(face_results.face_landmarks)

            if face_found:
                landmarks = face_results.face_landmarks[0]
                face_lost_at = None
                drowsy_at_face_loss = False
                face_missing_alert_repeats = 0
                recalibration_requested = False

                left_ear = eye_aspect_ratio(landmarks, config.LEFT_EYE_IDX, w, h)
                right_ear = eye_aspect_ratio(landmarks, config.RIGHT_EYE_IDX, w, h)
                current_ear = (left_ear + right_ear) / 2.0
                current_mar = mouth_aspect_ratio(
                    landmarks,
                    config.MOUTH_TOP, config.MOUTH_BOTTOM,
                    config.MOUTH_LEFT, config.MOUTH_RIGHT,
                    w, h,
                )
                x1, y1, x2, y2 = padded_face_box(landmarks, w, h, config.CIGARETTE_FACE_PADDING)
                face_crop = frame[y1:y2, x1:x2]

                # Smooth MAR so a single noisy frame at the threshold can't flip mouth-open state.
                smoothed_mar = current_mar if smoothed_mar is None else (
                    config.MAR_SMOOTHING_ALPHA * current_mar
                    + (1 - config.MAR_SMOOTHING_ALPHA) * smoothed_mar
                )
                left_angle, right_angle = compute_corner_lift_angles(landmarks, w, h)
                avg_lift_angle = (left_angle + right_angle) / 2.0

                smoothed_lift_angle = avg_lift_angle if smoothed_lift_angle is None else (
                    config.MAR_SMOOTHING_ALPHA * avg_lift_angle + (1 - config.MAR_SMOOTHING_ALPHA) * smoothed_lift_angle
                )

                if face_results.facial_transformation_matrixes:
                    pose_angles = head_pose_angles(face_results.facial_transformation_matrixes[0])
                    if pose_angles:
                        current_yaw_deg, current_pitch_deg, current_roll_deg = pose_angles

                # Background calibration: startup timed out, or a long face loss
                # suggested a driver change. Defaults stay in use until it completes.
                # Same quality gate as startup calibration (face fully visible,
                # eyes open, roughly forward, not covered), so a bad frame can't
                # skew the new baseline.
                if pending_calibrator is not None and not camera_monitor.suspect and _face_problem(
                        landmarks, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), w, h,
                        left_ear, right_ear, current_yaw_deg, current_pitch_deg) is None:
                    pending_calibrator.add(current_ear, current_yaw_deg, current_pitch_deg, current_roll_deg)
                    if pending_calibrator.done:
                        baseline = pending_calibrator.result()
                        pending_calibrator = None
                        baseline_ear = baseline.ear
                        ear_threshold = baseline.ear_threshold
                        gaze_baseline_yaw, gaze_baseline_pitch, gaze_baseline_roll = (
                            baseline.yaw, baseline.pitch, baseline.roll
                        )
                        blink_monitor.blink_ear_threshold = ear_threshold
                        print("[INFO] Background calibration complete")
                        _print_baseline(baseline)
                        if identity is not None and identity.phase == "confirmed":
                            profiles.save(identity.driver_id, baseline)
                        log_alert("calibration_updated", {
                            "baseline_ear": round(baseline.ear, 3),
                            "ear_threshold": round(baseline.ear_threshold, 3),
                            "clamped": baseline.clamped,
                            "yaw_deg": round(baseline.yaw, 1),
                            "pitch_deg": round(baseline.pitch, 1),
                            "roll_deg": round(baseline.roll, 1),
                        })

                # EAR is only trustworthy near the calibrated forward pose: looking
                # down narrows the eye (false closure), turning foreshortens one eye.
                ear_pose_reliable = (
                    abs(current_yaw_deg - gaze_baseline_yaw) <= config.EAR_POSE_YAW_MAX
                    and abs(current_pitch_deg - gaze_baseline_pitch) <= config.EAR_POSE_PITCH_MAX
                )

                # Driver recognition: only on frames that pass the calibration
                # quality check (whole face visible, eyes open, roughly forward).
                # The frame is mirrored, so identify_landmarks un-mirrors it to
                # match the (unmirrored) enrolment photos.
                if identity is not None:
                    face_bbox = padded_face_box(landmarks, w, h, 0.1)
                if identity is not None and not camera_monitor.suspect and identity.wants_sample(now):
                    if _face_problem(landmarks, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), w, h,
                                     left_ear, right_ear, current_yaw_deg, current_pitch_deg) is None:
                        id_status, id_emp, id_name, id_score, face_bbox = driver_identifier.identify_landmarks(
                            frame, landmarks, w, h, mirrored=True)
                        identity_events = identity.update(now, id_status, id_emp, id_name, id_score)

                looking_away = (
                    abs(current_yaw_deg - gaze_baseline_yaw) > config.YAW_ANGLE_MAX
                    or abs(current_pitch_deg - gaze_baseline_pitch) > config.PITCH_ANGLE_MAX
                    or abs(current_roll_deg - gaze_baseline_roll) > config.ROLL_ANGLE_MAX
                )
                if looking_away:
                    if gaze_away_start is None:
                        gaze_away_start = now
                else:
                    gaze_away_start = None

                current_pitch_ratio = head_pitch_ratio(
                    landmarks, config.FOREHEAD_IDX, config.CHIN_IDX, config.NOSE_TIP_IDX
                )
                # Smooth so a single noisy landmark frame can't register as a "sudden" drop.
                smoothed_pitch_ratio = current_pitch_ratio if smoothed_pitch_ratio is None else (
                    config.HEAD_DROP_SMOOTHING_ALPHA * current_pitch_ratio
                    + (1 - config.HEAD_DROP_SMOOTHING_ALPHA) * smoothed_pitch_ratio
                )
                pitch_history.append((now, smoothed_pitch_ratio))
                while pitch_history and now - pitch_history[0][0] > config.HEAD_DROP_WINDOW_SEC:
                    pitch_history.popleft()

                is_head_down = smoothed_pitch_ratio > config.PITCH_RATIO_DOWN
                head_down_sec = head_down_sec + frame_dt if is_head_down else max(0.0, head_down_sec - frame_dt)

                # Smooth EAR so a single noisy frame can't flip the closure timer.
                # Time-based alpha keeps the lag constant in seconds at any fps.
                if smoothed_ear is None:
                    smoothed_ear = current_ear
                else:
                    ear_alpha = _ema_alpha(frame_dt, config.EAR_SMOOTHING_TAU_SEC)
                    smoothed_ear = ear_alpha * current_ear + (1 - ear_alpha) * smoothed_ear
                eyes_occluded = blink_monitor.update(now, current_ear)

                # Both eyes must agree they're closed -- rejects winks. Relative to
                # this driver's baseline so narrow/wide eyes get the same tolerance.
                eyes_agree = abs(left_ear - right_ear) < config.EAR_ASYMMETRY_RATIO * baseline_ear
                eyes_closed = smoothed_ear < ear_threshold and eyes_agree

                # Off-pose frames HOLD the timer and skip PERCLOS rather than feed
                # them an unreliable EAR. A real nod-off is caught by head-drop.
                if ear_pose_reliable:
                    eyes_closed_sec = eyes_closed_sec + frame_dt if eyes_closed else max(0.0, eyes_closed_sec - frame_dt)
                    # Rolling % of the last PERCLOS_WINDOW_SEC spent with eyes closed (time-weighted).
                    perclos_monitor.update(now, eyes_closed, frame_dt)
                else:
                    perclos_monitor.evict(now)
                current_perclos = perclos_monitor.value
                perclos_ready = perclos_monitor.ready

                # Count mouth-open rising edges in the trailing window to catch talking/laughing/singing.
                mouth_open = smoothed_mar > config.MAR_THRESHOLD
                is_open_raw = current_mar > config.MAR_THRESHOLD
                mouth_open_history.append((now, is_open_raw))
                while mouth_open_history and now - mouth_open_history[0][0] > config.OSCILLATION_WINDOW_SEC:
                    mouth_open_history.popleft()

                rising_edges = sum(
                    1 for i in range(1, len(mouth_open_history))
                    if mouth_open_history[i][1] and not mouth_open_history[i - 1][1]
                )
                is_oscillating = rising_edges > config.YAWN_MAX_TRANSITIONS

                # A yawn is one continuous mouth-open stretch; oscillation restarts the hold.
                if mouth_open:
                    if mouth_open_start is None:
                        mouth_open_start = now
                        mar_during_hold = []
                    held_sec = now - mouth_open_start
                    duration_ok = held_sec >= config.YAWN_HOLD_SEC

                    symmetry_ok = (not config.YAWN_REQUIRE_SYMMETRIC) or (smoothed_lift_angle < config.CORNER_LIFT_MAX)
                    mar_during_hold.append(current_mar)
                    mar_variance_ok = (max(mar_during_hold) - min(mar_during_hold)) < config.YAWN_MAX_MAR_VARIANCE

                    yawn_confirmed = duration_ok and symmetry_ok and mar_variance_ok and not is_oscillating

                    if is_oscillating:
                        mouth_open_start = now
                        mar_during_hold = []
                else:
                    mouth_open_start = None
                    mar_during_hold = []
                    yawn_confirmed = False

            else:
                # No face (or camera blocked). Don't log the last seen pose as if it were current.
                current_yaw_deg = current_pitch_deg = current_roll_deg = 0.0
                current_pitch_ratio = 0.5
                eyes_closed = False
                if face_lost_at is None:
                    face_lost_at = now
                    # Snapshot whether the driver looked drowsy right before the face
                    # vanished -- a slumping driver can drop out of frame entirely.
                    # (Not when the camera got blocked: that has its own alert.)
                    drowsy_at_face_loss = not camera_blocked and (
                        eyes_closed_sec >= 0.5 * config.EYES_CLOSED_HOLD_SEC or is_head_down
                    )
                face_lost_sec = now - face_lost_at
                if identity is not None:
                    identity.face_lost(face_lost_sec)
                # Tolerate brief tracking loss before decaying counters;
                # a blocked camera resets everything immediately.
                if camera_blocked or now - face_lost_at > config.NO_FACE_GRACE_SEC:
                    eyes_closed_sec = max(0.0, eyes_closed_sec - frame_dt)
                    smoothed_ear = None
                    mouth_open_start = None
                    smoothed_mar = None
                    smoothed_lift_angle = None
                    mar_during_hold = []
                    mouth_open_history.clear()
                    yawn_confirmed = False
                    last_yawn_confirmed = False
                    gaze_away_start = None
                    head_down_sec = max(0.0, head_down_sec - frame_dt)
                    pitch_history.clear()
                    blink_monitor.reset()
                    eyes_occluded = False
                # Pause PERCLOS during brief face loss (old samples still age out);
                # only clear it after a long absence.
                perclos_monitor.evict(now)
                if face_lost_sec > config.PERCLOS_RESET_AFTER_FACE_LOSS_SEC:
                    perclos_monitor.reset()
                current_perclos = perclos_monitor.value
                # Long absence = possible driver change: recalibrate when a face returns.
                if (
                    config.RECALIBRATE_AFTER_FACE_LOSS_SEC > 0
                    and config.EAR_CALIBRATION_FRAMES > 0
                    and face_lost_sec > config.RECALIBRATE_AFTER_FACE_LOSS_SEC
                    and not recalibration_requested
                ):
                    recalibration_requested = True
                    pending_calibrator = BaselineCalibrator(config.EAR_CALIBRATION_FRAMES)
                    print("[INFO] Face absent for a long time -- will recalibrate when a face returns")

            # Keep the clear-view reference up to date (slowly) while the face is visible.
            if face_found and not camera_monitor.suspect:
                camera_monitor.update_reference()

            # A long camera block can hide a driver swap just like face loss can.
            blocked_since = (blocked_since or now) if camera_blocked else None
            if identity is not None and blocked_since is not None:
                identity.face_lost(now - blocked_since)

            # ── Driver face not detected (camera itself is clear) ──
            if camera_blocked:
                face_lost_at = now      # count "face missing" only from when the view is clear again
            face_missing_sec = (now - face_lost_at) if (face_lost_at is not None and not camera_blocked) else 0.0
            if face_missing_sec >= no_face_alert_sec and now - last_no_face_alert > no_face_repeat_sec:
                no_face_alert_count += 1
                print(f"[NO FACE #{no_face_alert_count}] Driver face not detected for {face_missing_sec:.1f}s")
                play_occlusion_alert()
                log_alert("driver_not_detected", {"missing_sec": round(face_missing_sec, 1)})
                last_no_face_alert = now

            # ── Driver recognition events ──
            # Tag everything from here on (incl. these events) with the decided driver.
            if identity is not None and identity_events:
                set_alert_context(**identity.context())
            for ev, info in identity_events:
                if ev == "driver_identified":
                    print(f"[DRIVER] Identified: {identity.label} (score {info['score']:.2f}, votes {info['votes']})")
                    log_alert("driver_identified", info)
                    if info["driver_id"] != UNKNOWN:
                        if baseline.calibrated and pending_calibrator is None:
                            profiles.save(info["driver_id"], baseline)   # startup calibration was theirs
                        elif not baseline.calibrated and profiles.get(info["driver_id"]):
                            ev = "apply_profile"                          # startup fell back to defaults
                elif ev == "driver_changed":
                    print(f"[DRIVER] Changed: {info['previous_driver_id']} -> {identity.label}")
                    log_alert("driver_changed", info)
                    # New person: previous driver's eye history must not carry over.
                    perclos_monitor.reset()
                    blink_monitor.reset()
                    eyes_closed_sec = 0.0
                    if config.EAR_CALIBRATION_FRAMES > 0:
                        pending_calibrator = BaselineCalibrator(config.EAR_CALIBRATION_FRAMES)
                    if info["driver_id"] != UNKNOWN and profiles.get(info["driver_id"]):
                        ev = "apply_profile"
                elif ev == "unknown_driver" and config.DRIVER_ID_ALERT_UNKNOWN:
                    unknown_alert_count += 1
                    print(f"[UNKNOWN DRIVER #{unknown_alert_count}] Driver is not an enrolled employee "
                          f"(best score {info['score']:.2f})")
                    play_unknown_driver_alert()
                    log_alert("unknown_driver", info)
                    last_unknown_alert = now

                if ev == "apply_profile":
                    # Known driver with a saved baseline: use it now instead of
                    # defaults / the previous driver's; background calibration refreshes it.
                    baseline = profiles.get(info["driver_id"])
                    baseline_ear, ear_threshold = baseline.ear, baseline.ear_threshold
                    gaze_baseline_yaw, gaze_baseline_pitch, gaze_baseline_roll = baseline.yaw, baseline.pitch, baseline.roll
                    blink_monitor.blink_ear_threshold = ear_threshold
                    print(f"[DRIVER] Loaded saved profile for {identity.label}")
                    _print_baseline(baseline)
                    log_alert("calibration_updated", {
                        "source": "driver_profile",
                        "baseline_ear": round(baseline.ear, 3),
                        "ear_threshold": round(baseline.ear_threshold, 3),
                        "yaw_deg": round(baseline.yaw, 1),
                        "pitch_deg": round(baseline.pitch, 1),
                        "roll_deg": round(baseline.roll, 1),
                    })
            if identity is not None:
                set_alert_context(**identity.context())
                # Unknown driver still at the wheel: repeat the alert periodically.
                if (config.DRIVER_ID_ALERT_UNKNOWN and identity.driver_id == UNKNOWN
                        and identity.phase == "confirmed" and face_found
                        and now - last_unknown_alert > config.DRIVER_ID_UNKNOWN_REPEAT_SEC):
                    unknown_alert_count += 1
                    print(f"[UNKNOWN DRIVER #{unknown_alert_count}] still driving")
                    play_unknown_driver_alert()
                    log_alert("unknown_driver", {"repeat": True, "score": round(identity.score, 3)})
                    last_unknown_alert = now

            # ── Drowsiness alert (eyes) ──
            # A current closed reading is never suppressed (blink_visibility also
            # enforces this): occluding lenses read as OPEN, not closed.
            ear_suppressed = eyes_occluded and not eyes_closed
            if eyes_closed_sec >= config.EYES_CLOSED_HOLD_SEC and not ear_suppressed and now - last_alert > config.ALERT_COOLDOWN_SEC:
                alert_count += 1
                print(f"[ALERT #{alert_count}] Drowsiness detected EAR={current_ear:.3f} "
                      f"smoothed={smoothed_ear or 0.0:.3f} threshold={ear_threshold:.3f}")
                play_alert()
                log_alert("drowsiness_detected", {
                    "ear": round(current_ear, 3),
                    "smoothed_ear": round(smoothed_ear or 0.0, 3),
                    "ear_threshold": round(ear_threshold, 3),
                    "eyes_closed_sec": round(eyes_closed_sec, 2),
                    "yaw_deg": round(current_yaw_deg, 1),
                    "pitch_deg": round(current_pitch_deg, 1),
                })
                last_alert = now

            # ── Driver not visible after looking drowsy (slumped out of frame) ──
            if (
                face_lost_at is not None
                and not camera_blocked
                and drowsy_at_face_loss
                and now - face_lost_at >= config.FACE_MISSING_ALERT_SEC
                and face_missing_alert_repeats < config.FACE_MISSING_ALERT_MAX_REPEATS
                and now - last_alert > config.ALERT_COOLDOWN_SEC
            ):
                face_missing_alert_repeats += 1
                alert_count += 1
                print(f"[ALERT #{alert_count}] Driver not visible after drowsy signs "
                      f"({now - face_lost_at:.1f}s)")
                play_alert()
                log_alert("driver_not_visible", {
                    "face_missing_sec": round(now - face_lost_at, 1),
                    "repeat": face_missing_alert_repeats,
                })
                last_alert = now

            # ── Eye-occlusion alert (eyes hidden from camera, e.g. sunglasses) ──
            if eyes_occluded and now - last_occlusion_alert > config.OCCLUSION_ALERT_REPEAT_SEC:
                occlusion_alert_count += 1
                print(f"[OCCLUSION #{occlusion_alert_count}] Eyes hidden from camera EAR={current_ear:.3f}")
                play_occlusion_alert()
                log_alert("eyes_occluded", {"ear": round(current_ear, 3)})
                last_occlusion_alert = now

            # ── PERCLOS alert (rolling % eye closure) ──
            # perclos_ready requires enough window coverage, so one closed frame
            # after startup/reset can't read as 100%.
            if perclos_ready and current_perclos >= config.PERCLOS_ALERT_THRESHOLD and now - last_perclos_alert > config.PERCLOS_COOLDOWN_SEC:
                perclos_alert_count += 1
                print(f"[PERCLOS #{perclos_alert_count}] Rolling eye closure {current_perclos:.0%}")
                play_perclos_alert()
                log_alert("perclos_high", {
                    "perclos": round(current_perclos, 3),
                    "coverage_sec": round(perclos_monitor.coverage_sec, 1),
                    "ear_threshold": round(ear_threshold, 3),
                })
                last_perclos_alert = now

            # ── Yawn alert (rising edge only) ──
            if yawn_confirmed and not last_yawn_confirmed and now - last_yawn_time > config.YAWN_COOLDOWN_SEC:
                yawn_count += 1
                print(f"[YAWN #{yawn_count}] Yawn detected MAR={current_mar:.3f}")
                play_yawn_alert()
                log_alert("yawn_detected", {"mar": round(current_mar, 3)})
                last_yawn_time = now
            last_yawn_confirmed = yawn_confirmed

            # ── Distraction alert (sustained look-away) ──
            gaze_away_elapsed = (now - gaze_away_start) if gaze_away_start is not None else 0.0
            if gaze_away_elapsed >= config.DISTRACTION_HOLD_SEC and now - last_gaze_alert > config.DISTRACTION_COOLDOWN_SEC:
                gaze_distraction_count += 1
                print(f"[DISTRACTION #{gaze_distraction_count}] Looking away yaw={current_yaw_deg:.1f} pitch={current_pitch_deg:.1f} roll={current_roll_deg:.1f}")
                play_distraction_alert()
                log_alert("distraction_detected", {
                    "source": "gaze_away",
                    "yaw_deg": round(current_yaw_deg, 1),
                    "pitch_deg": round(current_pitch_deg, 1),
                    "roll_deg": round(current_roll_deg, 1),
                })
                last_gaze_alert = now

            # ── Head drop alert (sudden nod, held down) ──
            head_drop_hold_sec = (
                config.OCCLUSION_HEAD_DROP_HOLD_SEC if eyes_occluded else config.HEAD_DROP_HOLD_SEC
            )
            window_full = bool(pitch_history) and (now - pitch_history[0][0]) >= config.HEAD_DROP_WINDOW_SEC * 0.9
            pitch_rise = (smoothed_pitch_ratio - min(v for _, v in pitch_history)) if window_full else 0.0
            head_drop_confirmed = head_down_sec >= head_drop_hold_sec and pitch_rise >= config.HEAD_DROP_DELTA
            if head_drop_confirmed and now - last_head_drop_alert > config.HEAD_DROP_COOLDOWN_SEC:
                head_drop_count += 1
                print(f"[HEAD DROP #{head_drop_count}] pitch_ratio={current_pitch_ratio:.3f} rise={pitch_rise:.3f}")
                play_head_drop_alert()
                log_alert("head_drop_detected", {"pitch_ratio": round(current_pitch_ratio, 3), "pitch_rise": round(pitch_rise, 3)})
                last_head_drop_alert = now

            # ── YOLO detectors: skipped while the camera is blocked ──
            # (a black frame would otherwise read as "no seatbelt" and fire false alerts)
            phone_detected = False
            phone_confidence = 0.0
            if phone_detector and not camera_blocked:
                phone_detected, phone_confidence, phone_alert_fired, phone_distraction_fired = phone_detector.process(frame, now)
                if phone_alert_fired:
                    print(f"[PHONE #{phone_detector.alert_count}] Phone detected conf={phone_confidence:.3f}")
                    play_phone_alert()
                if phone_distraction_fired:
                    print(f"[DISTRACTION #{phone_detector.distraction_count}] Possible phone (low confidence) conf={phone_confidence:.3f}")
                    play_distraction_alert()

            cigarette_detected = False
            cigarette_confidence = 0.0
            if cigarette_detector and not camera_blocked:
                cigarette_detected, cigarette_confidence, cigarette_alert_fired = cigarette_detector.process(face_crop, now)
                if cigarette_alert_fired:
                    print(f"[CIGARETTE #{cigarette_detector.alert_count}] Cigarette detected conf={cigarette_confidence:.3f}")
                    play_cigarette_alert()

            seatbelt_present = True
            seatbelt_confidence = 0.0
            if seatbelt_detector and not camera_blocked:
                seatbelt_present, seatbelt_confidence, seatbelt_alert_fired = seatbelt_detector.process(frame, now)
                if seatbelt_alert_fired:
                    print(f"[SEATBELT #{seatbelt_detector.alert_count}] No seatbelt detected conf={seatbelt_confidence:.3f}")
                    play_seatbelt_alert()

            frame_elapsed = time.perf_counter() - frame_start
            current_fps = 1.0 / frame_elapsed if frame_elapsed > 0 else 0.0
            frame_log.write(
                frame_num, face_found, current_ear, current_mar, phone_confidence,
                cigarette_confidence, seatbelt_confidence, current_perclos,
                current_yaw_deg, current_pitch_deg, current_roll_deg, current_pitch_ratio,
                current_fps,
                smoothed_ear=smoothed_ear or 0.0,
                ear_threshold=ear_threshold,
                eyes_closed=eyes_closed,
                eyes_closed_sec=eyes_closed_sec,
                ear_pose_reliable=ear_pose_reliable,
                eyes_occluded=eyes_occluded,
                perclos_ready=perclos_ready,
                driver_id=(identity.driver_id or "") if identity is not None else "",
            )

            # ── Display ──
            if display_on:
                is_drowsy = eyes_closed_sec >= config.EYES_CLOSED_HOLD_SEC
                is_yawning = yawn_confirmed

                GREEN = (0, 255, 0)
                ORANGE = (0, 165, 255)
                RED = (0, 0, 255)

                is_distracted = gaze_away_elapsed >= config.DISTRACTION_HOLD_SEC
                is_perclos_high = perclos_monitor.ready and current_perclos >= config.PERCLOS_ALERT_THRESHOLD
                total_distractions = gaze_distraction_count + (phone_detector.distraction_count if phone_detector else 0)

                occlusion_text = " [EYES OCCLUDED]" if eyes_occluded else ""
                if face_results.face_landmarks and not ear_pose_reliable:
                    occlusion_text += " [OFF-POSE]"
                ear_text = f"EAR: {current_ear:.3f} ({eyes_closed_sec:.2f}s/{config.EYES_CLOSED_HOLD_SEC:.2f}s)  Alerts:{alert_count}{occlusion_text}"
                mar_text = f"MAR: {current_mar:.3f} ({'YAWN' if is_yawning else 'OK'})  Yawns:{yawn_count}"
                yaw_text = f"Yaw:{current_yaw_deg:.0f} Pitch:{current_pitch_deg:.0f} Roll:{current_roll_deg:.0f} ({gaze_away_elapsed:.1f}s/{config.DISTRACTION_HOLD_SEC:.1f}s)  Distractions:{total_distractions}"
                pitch_text = f"Pitch: {current_pitch_ratio:.2f} ({head_down_sec:.2f}s/{head_drop_hold_sec:.2f}s)  HeadDrops:{head_drop_count}"
                perclos_warmup = "" if perclos_monitor.ready else " warming up"
                perclos_text = f"PERCLOS: {current_perclos:.0%} ({config.PERCLOS_WINDOW_SEC:.0f}s{perclos_warmup})  Alerts:{perclos_alert_count}"
                fps_text = f"FPS: {current_fps:.1f}"

                cv2.putText(frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE if is_drowsy else GREEN, 2)
                cv2.putText(frame, mar_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_yawning else GREEN, 2)
                cv2.putText(frame, yaw_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_distracted else GREEN, 2)
                cv2.putText(frame, pitch_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_head_down else GREEN, 2)
                cv2.putText(frame, perclos_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_perclos_high else GREEN, 2)
                cv2.putText(frame, fps_text, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

                if phone_detector:
                    phone_text = f"Phone: {phone_confidence:.2f}  Alerts:{phone_detector.alert_count}"
                    cv2.putText(frame, phone_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if phone_detected else GREEN, 2)

                if cigarette_detector:
                    cigarette_text = f"Cigarette: {cigarette_confidence:.2f}  Alerts:{cigarette_detector.alert_count}"
                    cv2.putText(frame, cigarette_text, (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if cigarette_detected else GREEN, 2)

                if seatbelt_detector:
                    seatbelt_text = f"Seatbelt: {seatbelt_confidence:.2f}  Alerts:{seatbelt_detector.alert_count}"
                    cv2.putText(frame, seatbelt_text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN if seatbelt_present else RED, 2)

                if identity is not None:
                    id_color = (GREEN if identity.driver_id not in (None, UNKNOWN)
                                else RED if identity.driver_id == UNKNOWN else (0, 200, 255))
                    cv2.putText(frame, f"Driver: {identity.label}", (10, 255),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, id_color, 2)
                    if face_bbox is not None:
                        driver_identifier.draw_face_box(
                            frame, face_bbox, "KNOWN" if identity.driver_id not in (None, UNKNOWN) else "UNKNOWN",
                            identity.driver_id or "", identity.driver_name, identity.score)

                # Drawn last so it sits on top of everything else.
                if camera_blocked:
                    cv2.putText(frame, f"CAMERA BLOCKED ({camera_monitor.reason})", (10, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3)
                    cv2.putText(frame, "Detection paused - uncover the camera", (10, h // 2 + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
                elif face_missing_sec >= no_face_alert_sec:
                    cv2.putText(frame, "DRIVER FACE NOT DETECTED", (10, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3)

                if (w, h) != (config.DISPLAY_W, config.DISPLAY_H):
                    frame = cv2.resize(frame, (config.DISPLAY_W, config.DISPLAY_H))
                display.show(frame)

            # ── Keyboard ──
            key = keys.get_key()

            if key:
                key = key.lower()

                if key == "q":
                    print("[INFO] Quit")
                    break

                elif key == "v":
                    display_on = not display_on

                    if display_on:
                        display.start()
                        print("[INFO] Display ON")
                    else:
                        display.stop()
                        print("[INFO] Display OFF")

    finally:
        keys.restore()
        display.stop()
        grabber.release()
        cap.release()
        shutdown_audio()
        frame_log.close()

    # Real throughput -- frames processed over wall-clock time. Do NOT average
    # the per-frame fps column instead: a single near-instant frame (already
    # buffered by the camera) makes 1/frame_elapsed spike into the hundreds
    # and skews an arithmetic mean far above what the pipeline actually sustained.
    session_elapsed = time.monotonic() - session_start_time
    avg_fps = frame_num / session_elapsed if session_elapsed > 0 else 0.0

    phone_summary = f"  Phone alerts: {phone_detector.alert_count}" if phone_detector else ""
    cigarette_summary = f"  Cigarette alerts: {cigarette_detector.alert_count}" if cigarette_detector else ""
    seatbelt_summary = f"  Seatbelt alerts: {seatbelt_detector.alert_count}" if seatbelt_detector else ""
    total_distractions = gaze_distraction_count + (phone_detector.distraction_count if phone_detector else 0)
    print(
        f"[INFO] Session ended. Drowsiness alerts: {alert_count}  Yawns: {yawn_count}"
        f"  Distractions: {total_distractions}  Head drops: {head_drop_count}"
        f"  Occlusion alerts: {occlusion_alert_count}  PERCLOS alerts: {perclos_alert_count}"
        f"  Camera blocked alerts: {camera_block_alert_count}  No-face alerts: {no_face_alert_count}"
        f"{phone_summary}{cigarette_summary}{seatbelt_summary}"
    )
    if identity is not None:
        print(f"[INFO] Driver at end of session: {identity.label}  Unknown-driver alerts: {unknown_alert_count}")
    print(f"[INFO] Processed {frame_num} frames in {session_elapsed:.1f}s -- avg {avg_fps:.1f} fps")
    print(f"[INFO] Per-frame log saved to {frame_log.path}")