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
    play_head_drop_alert, shutdown_audio,
)
from .ear import eye_aspect_ratio
# from .mar import mouth_aspect_ratio
from .gaze import head_pose_angles, head_pitch_ratio
from .phone import PhoneDetector
from .display import LazyDisplay
from .keyboard_input import KeyReader


_shutdown_requested = False


def _handle_sigterm(signum, frame) -> None:
    """Sets a flag instead of exiting immediately, so the main loop's
    `finally` block still runs (releases camera, closes audio/log files)."""
    global _shutdown_requested
    _shutdown_requested = True


class MonotonicTimestamp:
    """Strictly increasing ms timestamps for MediaPipe's VIDEO mode, which
    requires each timestamp to exceed the last. Wall-clock time (time.time())
    can jump backward on NTP sync, which is common right after boot on a
    device with no RTC -- time.monotonic() never does."""

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

    if config.PHONE_DETECTION_ENABLED:
        param_file = os.path.join(config.PHONE_MODEL_PATH, "model.ncnn.param")
        bin_file = os.path.join(config.PHONE_MODEL_PATH, "model.ncnn.bin")

        if not (os.path.exists(param_file) and os.path.exists(bin_file)):
            print(f"[ERROR] Missing NCNN model files under {config.PHONE_MODEL_PATH}")
            print("Expected model.ncnn.param and model.ncnn.bin in that folder, or set "
                  "config.PHONE_DETECTION_ENABLED = False to run drowsiness/yawn only.")
            sys.exit(1)



def _get_jaw_open(face_results) -> float:
    if not face_results.face_blendshapes:
        return 0.0
    for b in face_results.face_blendshapes[0]:
        if b.category_name == "jawOpen":
            return b.score
    return 0.0

def _build_face_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.FaceLandmarker.create_from_options(options), mp


def _read_calibration_sample(cap, face_mesh, mp, timestamps: MonotonicTimestamp):
    """One capture+read cycle, reused by the calibration phase below.
    Returns (ear, yaw, pitch, roll) or None if no face was found."""
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    face_results = face_mesh.detect_for_video(mp_image, timestamps.next())

    if not face_results.face_landmarks:
        return None

    landmarks = face_results.face_landmarks[0]
    left_ear = eye_aspect_ratio(landmarks, config.LEFT_EYE_IDX, w, h)
    right_ear = eye_aspect_ratio(landmarks, config.RIGHT_EYE_IDX, w, h)
    ear = (left_ear + right_ear) / 2.0

    yaw, pitch, roll = 0.0, 0.0, 0.0
    if face_results.facial_transformation_matrixes:
        pose_angles = head_pose_angles(face_results.facial_transformation_matrixes[0])
        if pose_angles:
            yaw, pitch, roll = pose_angles

    return ear, yaw, pitch, roll


def _calibrate_baseline(cap, face_mesh, mp, timestamps: MonotonicTimestamp):
    """
    Sample EAR and head pose for config.EAR_CALIBRATION_FRAMES frames while
    the driver looks at the road normally, and derive:
      - a personal EAR threshold (eye shape varies per person)
      - a yaw/pitch/roll baseline ("forward" isn't 0 degrees unless the
        camera is mounted dead-center in front of the driver's face -- an
        A-pillar or off-center mount needs this offset subtracted out)

    Falls back to config.EAR_THRESHOLD and a zero gaze baseline if no face
    is found, or if none shows up within the timeout.
    """
    if config.EAR_CALIBRATION_FRAMES <= 0:
        return config.EAR_THRESHOLD, 0.0, 0.0, 0.0

    print(f"[INFO] Calibrating baseline ({config.EAR_CALIBRATION_FRAMES} frames, "
          "keep eyes open and look at the road normally)...")

    samples = []
    start_time = time.monotonic()
    while len(samples) < config.EAR_CALIBRATION_FRAMES:
        if time.monotonic() - start_time > config.EAR_CALIBRATION_TIMEOUT_SEC:
            print("[WARN] Calibration timed out -- falling back to defaults")
            return config.EAR_THRESHOLD, 0.0, 0.0, 0.0

        sample = _read_calibration_sample(cap, face_mesh, mp, timestamps)
        if sample is not None:
            samples.append(sample)

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


def run() -> None:
    config.apply_cpu_settings()

    cv2.ocl.setUseOpenCL(config.OPENCV_USE_OPENCL)
    cv2.setNumThreads(config.OPENCV_NUM_THREADS)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    _check_model()
    cleanup_old_logs()

    face_mesh, mp = _build_face_landmarker()
    phone_detector = PhoneDetector() if config.PHONE_DETECTION_ENABLED else None

    if config.RTSP_URL.strip():
        print(f"[INFO] Using RTSP stream: {config.RTSP_URL.strip()}")
    else:
        print(f"[INFO] Using local webcam (index {config.CAMERA_INDEX})")

    cap = _open_camera()

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    timestamps = MonotonicTimestamp()
    ear_threshold, gaze_baseline_yaw, gaze_baseline_pitch, gaze_baseline_roll = _calibrate_baseline(
        cap, face_mesh, mp, timestamps
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

    yawn_counter = 0
    # small_yawn_counter = 0
    yawn_count = 0
    last_yawn_alert = 0.0
    # Trailing "mouth open" history, used to tell one sustained yawn apart
    # from the repeated open/close of talking, laughing, or singing.
    mouth_open_history = deque(maxlen=config.YAWN_TRANSITION_WINDOW)

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
    # Trailing pitch history, used to check the drop happened quickly rather
    # than a slow, deliberate lean (e.g. checking a lap or console).
    pitch_history = deque(maxlen=max(2, int(config.CAM_FPS * config.HEAD_DROP_WINDOW_SEC)))

    frame_num = 0

    print()
    print("[INFO] Running")
    print("[INFO] Keys: v = toggle display, q = quit")

    try:
        while True:
            if _shutdown_requested:
                print("[INFO] Shutdown signal received")
                break

            frame_start = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                print("[WARN] Camera frame failed, attempting reconnect")
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

                continue

            frame_num += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ── Face landmarks -> EAR (drowsiness) + MAR (yawn) ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_results = face_mesh.detect_for_video(mp_image, timestamps.next())

            current_ear = 0.0
            current_mar = 0.0

            if face_results.face_landmarks:
                landmarks = face_results.face_landmarks[0]
                no_face_streak = 0

                left_ear = eye_aspect_ratio(landmarks, config.LEFT_EYE_IDX, w, h)
                right_ear = eye_aspect_ratio(landmarks, config.RIGHT_EYE_IDX, w, h)
                current_ear = (left_ear + right_ear) / 2.0
                # current_mar = mouth_aspect_ratio(
                #     landmarks,
                #     config.MOUTH_TOP, config.MOUTH_BOTTOM,
                #     config.MOUTH_LEFT, config.MOUTH_RIGHT,
                #     w, h,
                # )
                current_mar = _get_jaw_open(face_results)
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
                # Smooth so a single noisy/landmark-jitter frame can't
                # register on its own as a "sudden" drop.
                smoothed_pitch_ratio = current_pitch_ratio if smoothed_pitch_ratio is None else (
                    config.HEAD_DROP_SMOOTHING_ALPHA * current_pitch_ratio
                    + (1 - config.HEAD_DROP_SMOOTHING_ALPHA) * smoothed_pitch_ratio
                )
                pitch_history.append(smoothed_pitch_ratio)

                is_head_down = smoothed_pitch_ratio > config.PITCH_RATIO_DOWN
                head_drop_counter = head_drop_counter + 1 if is_head_down else max(0, head_drop_counter - 1)

                # Smooth EAR so a single noisy frame can't flip the counter.
                smoothed_ear = current_ear if smoothed_ear is None else (
                    config.EAR_SMOOTHING_ALPHA * current_ear
                    + (1 - config.EAR_SMOOTHING_ALPHA) * smoothed_ear
                )

                # Both eyes must agree they're closed -- rejects winks/side glances.
                eyes_agree = abs(left_ear - right_ear) < config.EAR_ASYMMETRY_MAX
                eyes_closed = smoothed_ear < ear_threshold and eyes_agree
                counter = counter + 1 if eyes_closed else max(0, counter - 1)

                # Yawn oscillation check: count mouth-open rising edges in the
                # trailing window to catch talking/laughing/singing.
                mouth_open = current_mar > config.MAR_THRESHOLD
                mouth_open_history.append(mouth_open)
                rising_edges = sum(
                    1 for i in range(1, len(mouth_open_history))
                    if mouth_open_history[i] and not mouth_open_history[i - 1]
                )
                is_oscillating = rising_edges > config.YAWN_MAX_TRANSITIONS

                if is_oscillating:
                    yawn_counter = 0
                else:
                    yawn_counter = yawn_counter + 1 if mouth_open else max(0, yawn_counter - 1)
            else:
                no_face_streak += 1
                # Tolerate brief tracking loss before decaying -- otherwise a
                # single dropped frame wipes out real progress toward an alert.
                if no_face_streak > config.NO_FACE_GRACE_FRAMES:
                    counter = max(0, counter - 1)
                    yawn_counter = max(0, yawn_counter - 1)
                    # small_yawn_counter = max(0, small_yawn_counter - 1)
                    gaze_away_start = None
                    head_drop_counter = max(0, head_drop_counter - 1)

            now = time.time()

            # ── Drowsiness alert (eyes) ──
            if counter >= config.CONSEC_FRAMES and now - last_alert > config.ALERT_COOLDOWN_SEC:
                alert_count += 1
                print(f"[ALERT #{alert_count}] Drowsiness detected EAR={current_ear:.3f}")
                play_alert()
                log_alert("drowsiness_detected", {"ear": round(current_ear, 3)})
                last_alert = now

            # ── Yawn alert (mouth): a clear big yawn OR a sustained small yawn ──
            yawn_confirmed = (
                yawn_counter >= config.YAWN_CONSEC_FRAMES
            )
            if yawn_confirmed and now - last_yawn_alert > config.YAWN_COOLDOWN_SEC:
                yawn_count += 1
                print(f"[YAWN #{yawn_count}] Yawn detected MAR={current_mar:.3f}")
                play_yawn_alert()
                log_alert("yawn_detected", {"mar": round(current_mar, 3)})
                last_yawn_alert = now

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
            pitch_rise = smoothed_pitch_ratio - min(pitch_history) if len(pitch_history) == pitch_history.maxlen else 0.0
            head_drop_confirmed = head_drop_counter >= config.HEAD_DROP_HOLD_FRAMES and pitch_rise >= config.HEAD_DROP_DELTA
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

            frame_elapsed = time.perf_counter() - frame_start
            current_fps = 1.0 / frame_elapsed if frame_elapsed > 0 else 0.0
            frame_log.write(frame_num, current_ear, current_mar, phone_confidence, current_fps)

            # ── Display ──
            if display_on:
                is_drowsy = counter >= config.CONSEC_FRAMES
                is_yawning = yawn_counter >= config.YAWN_CONSEC_FRAMES 

                GREEN = (0, 255, 0)
                ORANGE = (0, 165, 255)
                RED = (0, 0, 255)

                is_distracted = gaze_away_elapsed >= config.DISTRACTION_HOLD_SEC
                is_head_down = smoothed_pitch_ratio is not None and smoothed_pitch_ratio > config.PITCH_RATIO_DOWN
                total_distractions = gaze_distraction_count + (phone_detector.distraction_count if phone_detector else 0)

                ear_text = f"EAR: {current_ear:.3f} ({counter}/{config.CONSEC_FRAMES})  Alerts:{alert_count}"
                mar_text = f"JawOpen: {current_mar:.3f} ({yawn_counter}/{config.YAWN_CONSEC_FRAMES})  Yawns:{yawn_count}"
                yaw_text = f"Yaw:{current_yaw_deg:.0f} Pitch:{current_pitch_deg:.0f} Roll:{current_roll_deg:.0f} ({gaze_away_elapsed:.1f}s/{config.DISTRACTION_HOLD_SEC:.1f}s)  Distractions:{total_distractions}"
                pitch_text = f"Pitch: {current_pitch_ratio:.2f} ({head_drop_counter}/{config.HEAD_DROP_HOLD_FRAMES})  HeadDrops:{head_drop_count}"
                fps_text = f"FPS: {current_fps:.1f}"

                cv2.putText(frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE if is_drowsy else GREEN, 2)
                cv2.putText(frame, mar_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_yawning else GREEN, 2)
                cv2.putText(frame, yaw_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_distracted else GREEN, 2)
                cv2.putText(frame, pitch_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_head_down else GREEN, 2)
                cv2.putText(frame, fps_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

                if phone_detector:
                    phone_detector.draw(frame)
                    phone_text = f"Phone: {phone_confidence:.2f}  Alerts:{phone_detector.alert_count}"
                    cv2.putText(frame, phone_text, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if phone_detected else GREEN, 2)

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
        cap.release()
        shutdown_audio()
        frame_log.close()

    phone_summary = f"  Phone alerts: {phone_detector.alert_count}" if phone_detector else ""
    total_distractions = gaze_distraction_count + (phone_detector.distraction_count if phone_detector else 0)
    print(
        f"[INFO] Session ended. Drowsiness alerts: {alert_count}  Yawns: {yawn_count}"
        f"  Distractions: {total_distractions}  Head drops: {head_drop_count}{phone_summary}"
    )
    print(f"[INFO] Per-frame log saved to {frame_log.path}")