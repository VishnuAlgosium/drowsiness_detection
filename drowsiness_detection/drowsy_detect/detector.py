"""
detector.py
-----------
Ties together camera capture, MediaPipe FaceLandmarker (EAR/MAR), YOLO
phone detection, display, keyboard control, and audio/log alerting into
one shared run loop.
"""

import os
import sys
import time
from collections import deque

import cv2

from . import config
from .alerts import log_alert, FrameLogger
from .audio import play_alert, play_yawn_alert, play_phone_alert, play_distraction_alert, shutdown_audio
from .ear import eye_aspect_ratio
from .mar import mouth_aspect_ratio
from .gaze import head_yaw_ratio
from .phone import PhoneDetector
from .display import LazyDisplay
from .keyboard_input import KeyReader


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
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.FaceLandmarker.create_from_options(options), mp


def _read_ear_for_calibration(cap, face_mesh, mp):
    """One capture+EAR-read cycle, reused by the calibration phase below."""
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)
    face_results = face_mesh.detect_for_video(mp_image, timestamp_ms)

    if not face_results.face_landmarks:
        return None

    landmarks = face_results.face_landmarks[0]
    left_ear = eye_aspect_ratio(landmarks, config.LEFT_EYE_IDX, w, h)
    right_ear = eye_aspect_ratio(landmarks, config.RIGHT_EYE_IDX, w, h)
    return (left_ear + right_ear) / 2.0


def _calibrate_ear_threshold(cap, face_mesh, mp) -> float:
    """
    Sample EAR for config.EAR_CALIBRATION_FRAMES frames (assuming eyes are
    open) and derive a personal threshold, since a single fixed EAR cutoff
    doesn't fit everyone's eye shape. Falls back to config.EAR_THRESHOLD if
    no face is found.
    """
    if config.EAR_CALIBRATION_FRAMES <= 0:
        return config.EAR_THRESHOLD

    print(f"[INFO] Calibrating eye baseline ({config.EAR_CALIBRATION_FRAMES} frames, keep eyes open and face the camera)...")

    samples = []
    while len(samples) < config.EAR_CALIBRATION_FRAMES:
        ear = _read_ear_for_calibration(cap, face_mesh, mp)
        if ear is not None:
            samples.append(ear)

    if not samples:
        print("[WARN] Calibration found no face -- falling back to default EAR threshold")
        return config.EAR_THRESHOLD

    baseline_ear = sum(samples) / len(samples)
    ear_threshold = baseline_ear * config.EAR_THRESHOLD_RATIO
    print(f"[INFO] Baseline EAR={baseline_ear:.3f} -> using drowsy threshold={ear_threshold:.3f}")
    return ear_threshold


def run() -> None:
    config.apply_cpu_settings()

    cv2.ocl.setUseOpenCL(config.OPENCV_USE_OPENCL)
    cv2.setNumThreads(config.OPENCV_NUM_THREADS)

    _check_model()

    face_mesh, mp = _build_face_landmarker()
    phone_detector = PhoneDetector() if config.PHONE_DETECTION_ENABLED else None

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ear_threshold = _calibrate_ear_threshold(cap, face_mesh, mp)

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
    small_yawn_counter = 0
    yawn_count = 0
    last_yawn_alert = 0.0
    # Trailing "mouth open" history, used to tell one sustained yawn apart
    # from the repeated open/close of talking, laughing, or singing.
    mouth_open_history = deque(maxlen=config.YAWN_TRANSITION_WINDOW)

    gaze_counter = 0
    gaze_distraction_count = 0
    last_gaze_alert = 0.0
    current_yaw_ratio = 0.5

    frame_num = 0

    print()
    print("[INFO] Running")
    print("[INFO] Keys: v = toggle display, q = quit")

    try:
        while True:
            frame_start = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Camera frame failed")
                break

            frame_num += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ── Face landmarks -> EAR (drowsiness) + MAR (yawn) ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            face_results = face_mesh.detect_for_video(mp_image, timestamp_ms)

            current_ear = 0.0
            current_mar = 0.0

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
                current_yaw_ratio = head_yaw_ratio(
                    landmarks, config.NOSE_TIP_IDX, config.FACE_LEFT_EDGE_IDX, config.FACE_RIGHT_EDGE_IDX
                )
                looking_away = current_yaw_ratio < config.YAW_RATIO_LOW or current_yaw_ratio > config.YAW_RATIO_HIGH
                gaze_counter = gaze_counter + 1 if looking_away else max(0, gaze_counter - 1)

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
                mouth_open_loose = current_mar > config.MAR_LOW_THRESHOLD
                mouth_open_history.append(mouth_open_loose)
                rising_edges = sum(
                    1 for i in range(1, len(mouth_open_history))
                    if mouth_open_history[i] and not mouth_open_history[i - 1]
                )
                is_oscillating = rising_edges > config.YAWN_MAX_TRANSITIONS

                if is_oscillating:
                    yawn_counter = 0
                    small_yawn_counter = 0
                else:
                    yawn_counter = yawn_counter + 1 if current_mar > config.MAR_THRESHOLD else max(0, yawn_counter - 1)
                    small_yawn_counter = small_yawn_counter + 1 if mouth_open_loose else max(0, small_yawn_counter - 1)
            else:
                no_face_streak += 1
                # Tolerate brief tracking loss before decaying -- otherwise a
                # single dropped frame wipes out real progress toward an alert.
                if no_face_streak > config.NO_FACE_GRACE_FRAMES:
                    counter = max(0, counter - 1)
                    yawn_counter = max(0, yawn_counter - 1)
                    small_yawn_counter = max(0, small_yawn_counter - 1)
                    gaze_counter = max(0, gaze_counter - 1)

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
                or small_yawn_counter >= config.MAR_LOW_CONSEC_FRAMES
            )
            if yawn_confirmed and now - last_yawn_alert > config.YAWN_COOLDOWN_SEC:
                yawn_count += 1
                print(f"[YAWN #{yawn_count}] Yawn detected MAR={current_mar:.3f}")
                play_yawn_alert()
                log_alert("yawn_detected", {"mar": round(current_mar, 3)})
                last_yawn_alert = now

            # ── Distraction alert (sustained look-away) ──
            if gaze_counter >= config.DISTRACTION_CONSEC_FRAMES and now - last_gaze_alert > config.DISTRACTION_COOLDOWN_SEC:
                gaze_distraction_count += 1
                print(f"[DISTRACTION #{gaze_distraction_count}] Looking away yaw_ratio={current_yaw_ratio:.3f}")
                play_distraction_alert()
                log_alert("distraction_detected", {"source": "gaze_away", "yaw_ratio": round(current_yaw_ratio, 3)})
                last_gaze_alert = now

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
                is_yawning = yawn_counter >= config.YAWN_CONSEC_FRAMES or small_yawn_counter >= config.MAR_LOW_CONSEC_FRAMES

                GREEN = (0, 255, 0)
                ORANGE = (0, 165, 255)
                RED = (0, 0, 255)

                is_distracted = gaze_counter >= config.DISTRACTION_CONSEC_FRAMES
                total_distractions = gaze_distraction_count + (phone_detector.distraction_count if phone_detector else 0)

                ear_text = f"EAR: {current_ear:.3f} ({counter}/{config.CONSEC_FRAMES})  Alerts:{alert_count}"
                mar_text = f"MAR: {current_mar:.3f} ({yawn_counter}/{config.YAWN_CONSEC_FRAMES})  Yawns:{yawn_count}"
                yaw_text = f"Yaw: {current_yaw_ratio:.2f} ({gaze_counter}/{config.DISTRACTION_CONSEC_FRAMES})  Distractions:{total_distractions}"
                fps_text = f"FPS: {current_fps:.1f}"

                cv2.putText(frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE if is_drowsy else GREEN, 2)
                cv2.putText(frame, mar_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_yawning else GREEN, 2)
                cv2.putText(frame, yaw_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_distracted else GREEN, 2)
                cv2.putText(frame, fps_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

                if phone_detector:
                    phone_detector.draw(frame)
                    phone_text = f"Phone: {phone_confidence:.2f}  Alerts:{phone_detector.alert_count}"
                    cv2.putText(frame, phone_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if phone_detected else GREEN, 2)

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
        f"  Distractions: {total_distractions}{phone_summary}"
    )
    print(f"[INFO] Per-frame log saved to {frame_log.path}")