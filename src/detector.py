"""
detector.py
-----------
Ties together camera capture, MediaPipe FaceLandmarker (EAR/MAR), YOLO
phone detection, display, keyboard control, and audio/log alerting into
one shared run loop.
"""

import os
import signal
import sys
import time
from collections import deque

import cv2

from . import config
from .alerts import log_alert, FrameLogger, cleanup_old_logs
from .audio import (
    play_alert, play_yawn_alert, play_phone_alert, play_distraction_alert,
    play_head_drop_alert, play_occlusion_alert, play_perclos_alert,
    play_cigarette_alert, play_seatbelt_alert, shutdown_audio,
)
from .ear import eye_aspect_ratio, average_ear
from .mar import mouth_aspect_ratio
from .gaze import head_pose_angles, head_pitch_ratio
from .face_crop import padded_face_box
from .blink_visibility import BlinkVisibilityMonitor
from .perclos import PerclosMonitor
from .capture import FrameGrabber
from .phone import PhoneDetector
from .cigarette import CigaretteDetector
from .seatbelt import SeatbeltDetector
from .display import LazyDisplay
from .keyboard_input import KeyReader
import math


_shutdown_requested = False


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


def _read_calibration_sample(grabber: FrameGrabber, last_frame_id: int, face_mesh, mp, timestamps: MonotonicTimestamp):
    """One capture+read cycle, reused by the calibration phase below.
    Returns (sample, frame_id) where sample is (ear, yaw, pitch, roll) or
    None if no new frame was ready yet or no face was found."""
    frame, frame_id, _ = grabber.read()
    if frame is None or frame_id == last_frame_id:
        return None, last_frame_id

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    face_results = face_mesh.detect_for_video(mp_image, timestamps.next())

    if not face_results.face_landmarks:
        return None, frame_id

    landmarks = face_results.face_landmarks[0]
    ear = average_ear(landmarks, config.LEFT_EYE_IDX, config.RIGHT_EYE_IDX, w, h)

    yaw, pitch, roll = 0.0, 0.0, 0.0
    if face_results.facial_transformation_matrixes:
        pose_angles = head_pose_angles(face_results.facial_transformation_matrixes[0])
        if pose_angles:
            yaw, pitch, roll = pose_angles

    return (ear, yaw, pitch, roll), frame_id


def _calibrate_baseline(grabber: FrameGrabber, face_mesh, mp, timestamps: MonotonicTimestamp):
    """
    Sample EAR and head pose for config.EAR_CALIBRATION_FRAMES frames while
    the driver looks at the road normally, and derive:
      - a personal EAR threshold (eye shape varies per person)
      - a yaw/pitch/roll baseline (an off-center camera mount means
        "forward" isn't 0 degrees, so the offset needs subtracting out)

    Falls back to config.EAR_THRESHOLD and a zero gaze baseline if no face
    is found, or if none shows up within the timeout.
    """
    if config.EAR_CALIBRATION_FRAMES <= 0:
        return config.EAR_THRESHOLD, 0.0, 0.0, 0.0

    print(f"[INFO] Calibrating baseline ({config.EAR_CALIBRATION_FRAMES} frames, "
          "keep eyes open and look at the road normally)...")

    samples = []
    last_frame_id = -1
    start_time = time.monotonic()
    while len(samples) < config.EAR_CALIBRATION_FRAMES:
        if time.monotonic() - start_time > config.EAR_CALIBRATION_TIMEOUT_SEC:
            print("[WARN] Calibration timed out -- falling back to defaults")
            return config.EAR_THRESHOLD, 0.0, 0.0, 0.0

        sample, last_frame_id = _read_calibration_sample(grabber, last_frame_id, face_mesh, mp, timestamps)
        if sample is not None:
            samples.append(sample)
        else:
            time.sleep(0.002)  # no new frame yet; avoid busy-spinning

    if not samples:
        print("[WARN] Calibration found no face -- falling back to defaults")
        return config.EAR_THRESHOLD, 0.0, 0.0, 0.0

    ears, yaws, pitches, rolls = zip(*samples)

    baseline_ear = sum(ears) / len(ears)
    ear_threshold = baseline_ear * config.EAR_THRESHOLD_RATIO
    baseline_yaw = sum(yaws) / len(yaws)
    baseline_pitch = sum(pitches) / len(pitches)
    baseline_roll = sum(rolls) / len(rolls)

    print(f"[INFO] Baseline EAR={baseline_ear:.3f} -> drowsy threshold={ear_threshold:.3f}")
    print(f"[INFO] Baseline gaze yaw={baseline_yaw:.1f} pitch={baseline_pitch:.1f} roll={baseline_roll:.1f}")
    return ear_threshold, baseline_yaw, baseline_pitch, baseline_roll






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
    left_angle = math.degrees(math.atan2(-(ly - cy), cx - lx))    # note: left is to the LEFT of center, so dx = cx-lx (positive)
    right_angle = math.degrees(math.atan2(-(ry - cy), rx - cx))   # right corner: dx = rx-cx (positive)

    return left_angle, right_angle

def run() -> None:
    config.apply_cpu_settings()

    cv2.ocl.setUseOpenCL(config.OPENCV_USE_OPENCL)
    cv2.setNumThreads(config.OPENCV_NUM_THREADS)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    _check_model()
    cleanup_old_logs()

    face_mesh, mp = _build_face_landmarker()
    phone_detector = PhoneDetector() if config.PHONE_DETECTION_ENABLED else None
    cigarette_detector = CigaretteDetector() if config.CIGARETTE_DETECTION_ENABLED else None
    seatbelt_detector = SeatbeltDetector() if config.SEATBELT_DETECTION_ENABLED else None

    if config.RTSP_URL.strip():
        print(f"[INFO] Using RTSP stream: {config.RTSP_URL.strip()}")
    else:
        print(f"[INFO] Using local webcam (index {config.CAMERA_INDEX})")

    cap = _open_camera()

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    grabber = FrameGrabber(cap)

    timestamps = MonotonicTimestamp()
    ear_threshold, gaze_baseline_yaw, gaze_baseline_pitch, gaze_baseline_roll = _calibrate_baseline(
        grabber, face_mesh, mp, timestamps
    )

    display = LazyDisplay(config.DISPLAY_W, config.DISPLAY_H, config.CAM_FPS)
    display_on = config.DISPLAY_ON_START

    if display_on:
        display.start()

    keys = KeyReader()
    frame_log = FrameLogger()

    counter = 0
    alert_count = 0
    last_alert = 0.0
    smoothed_ear = None
    no_face_streak = 0
    eyes_occluded = False
    occlusion_alert_count = 0
    last_occlusion_alert = 0.0
    blink_monitor = BlinkVisibilityMonitor(
        config.NO_BLINK_TIMEOUT_SEC, ear_threshold, config.MAX_BLINK_FRAMES
    )

    perclos_monitor = PerclosMonitor(config.PERCLOS_WINDOW_SEC)
    current_perclos = 0.0
    perclos_alert_count = 0
    last_perclos_alert = 0.0

    smoothed_mar = None
    mouth_open_start = None   # wall-clock time MAR first crossed MAR_THRESHOLD, or None
    yawn_count = 0
    last_yawn_alert = 0.0
    last_yawn_confirmed = False   # previous frame's yawn_confirmed, to count on rising edge only
    mouth_open_history = deque()

    gaze_away_start = None
    gaze_away_elapsed = 0.0
    gaze_distraction_count = 0
    last_gaze_alert = 0.0
    current_yaw_deg = 0.0
    current_pitch_deg = 0.0
    current_roll_deg = 0.0

    head_drop_counter = 0
    head_drop_count = 0
    last_head_drop_alert = 0.0
    current_pitch_ratio = 0.5
    smoothed_pitch_ratio = None
    is_head_down = False
    pitch_history = deque()  # (timestamp, smoothed_pitch_ratio) pairs, evicted by wall-clock age
    
    last_yawn_time = -999.0
    smoothed_lift_angle = None
    yawn_confirmed = False
    
    
    held_sec = 0.0

    frame_num = 0
    last_frame_id = -1
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

            if frame_age_sec > config.CAMERA_STALE_FRAME_TIMEOUT_SEC:
                print("[WARN] Camera frame failed, attempting reconnect")
                grabber.release()
                cap.release()

                reconnected = False
                for attempt in range(1, config.CAMERA_RECONNECT_ATTEMPTS + 1):
                    time.sleep(config.CAMERA_RECONNECT_DELAY_SEC)
                    cap = _open_camera()
                    if cap.isOpened():
                        print(f"[INFO] Camera reconnected (attempt {attempt})")
                        reconnected = True
                        break

                if not reconnected:
                    print("[ERROR] Camera reconnect failed, exiting")
                    break

                grabber = FrameGrabber(cap)
                last_frame_id = -1
                continue

            if frame is None or frame_id == last_frame_id:
                time.sleep(0.002)  # no new frame yet; avoid busy-spinning
                continue
            last_frame_id = frame_id

            frame_start = time.perf_counter()
            now = time.time()

            frame_num += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_results = face_mesh.detect_for_video(mp_image, timestamps.next())

            current_ear = 0.0
            current_mar = 0.0
            face_crop = None

            if face_results.face_landmarks:
                landmarks = face_results.face_landmarks[0]
                no_face_streak = 0

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

                looking_away = (
                    abs(current_yaw_deg - gaze_baseline_yaw) > config.YAW_ANGLE_MAX
                    or abs(current_pitch_deg - gaze_baseline_pitch) > config.PITCH_ANGLE_MAX
                    or abs(current_roll_deg - gaze_baseline_roll) > config.ROLL_ANGLE_MAX
                )
                if looking_away:
                    if gaze_away_start is None:
                        gaze_away_start = time.time()
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
                head_drop_counter = head_drop_counter + 1 if is_head_down else max(0, head_drop_counter - 1)

                # Smooth EAR so a single noisy frame can't flip the counter.
                smoothed_ear = current_ear if smoothed_ear is None else (
                    config.EAR_SMOOTHING_ALPHA * current_ear
                    + (1 - config.EAR_SMOOTHING_ALPHA) * smoothed_ear
                )
                eyes_occluded = blink_monitor.update(now, current_ear)

                # Both eyes must agree they're closed -- rejects winks/side glances.
                eyes_agree = abs(left_ear - right_ear) < config.EAR_ASYMMETRY_MAX
                eyes_closed = smoothed_ear < ear_threshold and eyes_agree
                counter = counter + 1 if eyes_closed else max(0, counter - 1)

                # Rolling % of the last PERCLOS_WINDOW_SEC spent with eyes closed --
                # rises with frequent/long blinks, ahead of a single sustained closure.
                current_perclos = perclos_monitor.update(now, eyes_closed)

                # Count mouth-open rising edges in the trailing window to catch talking/laughing/singing.
                mouth_open = smoothed_mar > config.MAR_THRESHOLD
                is_open_raw = current_mar > config.MAR_THRESHOLD
                
                mouth_open_history.append((now, is_open_raw))
                
                
                while mouth_open_history and now - mouth_open_history[0][0] > config.OSCILLATION_WINDOW_SEC:
                    mouth_open_history.popleft()
            
                rising_edges = sum(
                    1 for i in range(1, len(mouth_open_history))
                    if mouth_open_history[i][1] and not mouth_open_history[i-1][1]
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
                        # Oscillation (talking/laughing) restarts the hold timer and its samples.
                        mouth_open_start = now
                        mar_during_hold = []
                else:
                    mouth_open_start = None
                    mar_during_hold = []
                    yawn_confirmed = False
                
            else:
                no_face_streak += 1
                # Tolerate brief tracking loss before decaying counters.
                if no_face_streak > config.NO_FACE_GRACE_FRAMES:
                    counter = max(0, counter - 1)
                    mouth_open_start = None
                    smoothed_mar = None
                    smoothed_lift_angle = None
                    mar_during_hold = []
                    mouth_open_history.clear()
                    yawn_confirmed = False
                    last_yawn_confirmed = False
                    gaze_away_start = None
                    head_drop_counter = max(0, head_drop_counter - 1)
                    pitch_history.clear()
                    blink_monitor.reset()
                    eyes_occluded = False
                    perclos_monitor.reset()
                    current_perclos = 0.0

            now = time.time()

            # ── Drowsiness alert (eyes) ──
            # Skipped while eyes_occluded: EAR is unreliable, so head-drop
            # (tightened below) becomes the primary fallback signal.
            if counter >= config.CONSEC_FRAMES and not eyes_occluded and now - last_alert > config.ALERT_COOLDOWN_SEC:
                alert_count += 1
                print(f"[ALERT #{alert_count}] Drowsiness detected EAR={current_ear:.3f}")
                play_alert()
                log_alert("drowsiness_detected", {"ear": round(current_ear, 3)})
                last_alert = now

            # ── Eye-occlusion alert (eyes hidden from camera, e.g. sunglasses) ──
            # Repeats periodically while occlusion persists, so a single
            # notice early in a long drive doesn't go unnoticed.
            if eyes_occluded and now - last_occlusion_alert > config.OCCLUSION_ALERT_REPEAT_SEC:
                occlusion_alert_count += 1
                print(f"[OCCLUSION #{occlusion_alert_count}] Eyes hidden from camera EAR={current_ear:.3f}")
                play_occlusion_alert()
                log_alert("eyes_occluded", {"ear": round(current_ear, 3)})
                last_occlusion_alert = now

            # ── PERCLOS alert (rolling % eye closure) ──
            # Catches fatigue building via frequent/long blinks, ahead of --
            # and independent of -- the sustained-closure counter above.
            if current_perclos >= config.PERCLOS_ALERT_THRESHOLD and now - last_perclos_alert > config.PERCLOS_COOLDOWN_SEC:
                perclos_alert_count += 1
                print(f"[PERCLOS #{perclos_alert_count}] Rolling eye closure {current_perclos:.0%}")
                play_perclos_alert()
                log_alert("perclos_high", {"perclos": round(current_perclos, 3)})
                last_perclos_alert = now

            # ── Yawn alert (mouth): sustained mouth-open, not a smile/talking ──
            # yawn_confirmed is computed above, in the per-frame face-detected block.
            # Count on rising edge only, so one long yawn doesn't retrigger every cooldown window.
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
            # Shorter hold required while eyes are occluded, since head-drop
            # is the primary fallback signal when EAR can't be trusted.
            head_drop_hold_frames = (
                config.OCCLUSION_HEAD_DROP_HOLD_FRAMES if eyes_occluded else config.HEAD_DROP_HOLD_FRAMES
            )
            window_full = bool(pitch_history) and (now - pitch_history[0][0]) >= config.HEAD_DROP_WINDOW_SEC * 0.9
            pitch_rise = (smoothed_pitch_ratio - min(v for _, v in pitch_history)) if window_full else 0.0
            head_drop_confirmed = head_drop_counter >= head_drop_hold_frames and pitch_rise >= config.HEAD_DROP_DELTA
            if head_drop_confirmed and now - last_head_drop_alert > config.HEAD_DROP_COOLDOWN_SEC:
                head_drop_count += 1
                print(f"[HEAD DROP #{head_drop_count}] pitch_ratio={current_pitch_ratio:.3f} rise={pitch_rise:.3f}")
                play_head_drop_alert()
                log_alert("head_drop_detected", {"pitch_ratio": round(current_pitch_ratio, 3), "pitch_rise": round(pitch_rise, 3)})
                last_head_drop_alert = now

            # ── Phone-use alert (+ low-confidence distraction) ──
            phone_detected = False
            phone_confidence = 0.0
            if phone_detector:
                phone_detected, phone_confidence, phone_alert_fired, phone_distraction_fired = phone_detector.process(frame, now)
                if phone_alert_fired:
                    print(f"[PHONE #{phone_detector.alert_count}] Phone detected conf={phone_confidence:.3f}")
                    play_phone_alert()
                if phone_distraction_fired:
                    print(f"[DISTRACTION #{phone_detector.distraction_count}] Possible phone (low confidence) conf={phone_confidence:.3f}")
                    play_distraction_alert()

            # ── Cigarette-use alert ──
            cigarette_detected = False
            cigarette_confidence = 0.0
            if cigarette_detector:
                cigarette_detected, cigarette_confidence, cigarette_alert_fired = cigarette_detector.process(face_crop, now)
                if cigarette_alert_fired:
                    print(f"[CIGARETTE #{cigarette_detector.alert_count}] Cigarette detected conf={cigarette_confidence:.3f}")
                    play_cigarette_alert()

            # ── Seatbelt alert (fires on sustained ABSENCE, not detection) ──
            seatbelt_present = True
            seatbelt_confidence = 0.0
            if seatbelt_detector:
                seatbelt_present, seatbelt_confidence, seatbelt_alert_fired = seatbelt_detector.process(frame, now)
                if seatbelt_alert_fired:
                    print(f"[SEATBELT #{seatbelt_detector.alert_count}] No seatbelt detected conf={seatbelt_confidence:.3f}")
                    play_seatbelt_alert()

            frame_elapsed = time.perf_counter() - frame_start
            current_fps = 1.0 / frame_elapsed if frame_elapsed > 0 else 0.0
            frame_log.write(
                frame_num, current_ear, current_mar, phone_confidence,
                cigarette_confidence, seatbelt_confidence, current_perclos,
                current_yaw_deg, current_pitch_deg, current_roll_deg, current_pitch_ratio,
                current_fps,
            )

            # ── Display ──
            if display_on:
                is_drowsy = counter >= config.CONSEC_FRAMES
                is_yawning = yawn_confirmed

                GREEN = (0, 255, 0)
                ORANGE = (0, 165, 255)
                RED = (0, 0, 255)

                is_distracted = gaze_away_elapsed >= config.DISTRACTION_HOLD_SEC
                is_perclos_high = current_perclos >= config.PERCLOS_ALERT_THRESHOLD
                total_distractions = gaze_distraction_count + (phone_detector.distraction_count if phone_detector else 0)

                occlusion_text = " [EYES OCCLUDED]" if eyes_occluded else ""
                ear_text = f"EAR: {current_ear:.3f} ({counter}/{config.CONSEC_FRAMES})  Alerts:{alert_count}{occlusion_text}"
                mar_text = f"MAR: {current_mar:.3f} ({'YAWN' if is_yawning else 'OK'})  Yawns:{yawn_count}"
                yaw_text = f"Yaw:{current_yaw_deg:.0f} Pitch:{current_pitch_deg:.0f} Roll:{current_roll_deg:.0f} ({gaze_away_elapsed:.1f}s/{config.DISTRACTION_HOLD_SEC:.1f}s)  Distractions:{total_distractions}"
                pitch_text = f"Pitch: {current_pitch_ratio:.2f} ({head_drop_counter}/{head_drop_hold_frames})  HeadDrops:{head_drop_count}"
                perclos_text = f"PERCLOS: {current_perclos:.0%} ({config.PERCLOS_WINDOW_SEC:.0f}s)  Alerts:{perclos_alert_count}"
                fps_text = f"FPS: {current_fps:.1f}"

                cv2.putText(frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE if is_drowsy else GREEN, 2)
                cv2.putText(frame, mar_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_yawning else GREEN, 2)
                cv2.putText(frame, yaw_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_distracted else GREEN, 2)
                cv2.putText(frame, pitch_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_head_down else GREEN, 2)
                cv2.putText(frame, perclos_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_perclos_high else GREEN, 2)
                cv2.putText(frame, fps_text, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

                if phone_detector:
                    # phone_detector.draw(frame)
                    phone_text = f"Phone: {phone_confidence:.2f}  Alerts:{phone_detector.alert_count}"
                    cv2.putText(frame, phone_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if phone_detected else GREEN, 2)

                if cigarette_detector:
                    cigarette_text = f"Cigarette: {cigarette_confidence:.2f}  Alerts:{cigarette_detector.alert_count}"
                    cv2.putText(frame, cigarette_text, (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if cigarette_detected else GREEN, 2)

                if seatbelt_detector:
                    # seatbelt_detector.draw(frame)
                    seatbelt_text = f"Seatbelt: {seatbelt_confidence:.2f}  Alerts:{seatbelt_detector.alert_count}"
                    cv2.putText(frame, seatbelt_text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN if seatbelt_present else RED, 2)

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
        f"{phone_summary}{cigarette_summary}{seatbelt_summary}"
    )
    print(f"[INFO] Processed {frame_num} frames in {session_elapsed:.1f}s -- avg {avg_fps:.1f} fps")
    print(f"[INFO] Per-frame log saved to {frame_log.path}")