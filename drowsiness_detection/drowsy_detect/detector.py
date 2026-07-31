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

import cv2

from . import config
from .alerts import log_alert, FrameLogger
from .audio import play_alert, play_yawn_alert, play_phone_alert, shutdown_audio
from .ear import average_ear
from .mar import mouth_aspect_ratio
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


def run() -> None:
    config.apply_cpu_settings()

    cv2.ocl.setUseOpenCL(config.OPENCV_USE_OPENCL)
    cv2.setNumThreads(config.OPENCV_NUM_THREADS)

    _check_model()

    face_mesh, mp = _build_face_landmarker()
    phone_detector = PhoneDetector() if config.PHONE_DETECTION_ENABLED else None

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    display = LazyDisplay(config.DISPLAY_W, config.DISPLAY_H, config.CAM_FPS)
    display_on = config.DISPLAY_ON_START

    if display_on:
        display.start()

    keys = KeyReader()
    frame_log = FrameLogger()

    counter = 0
    alert_count = 0
    last_alert = 0.0

    yawn_counter = 0
    yawn_count = 0
    last_yawn_alert = 0.0

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

                current_ear = average_ear(landmarks, config.LEFT_EYE_IDX, config.RIGHT_EYE_IDX, w, h)
                current_mar = mouth_aspect_ratio(
                    landmarks,
                    config.MOUTH_TOP, config.MOUTH_BOTTOM,
                    config.MOUTH_LEFT, config.MOUTH_RIGHT,
                    w, h,
                )

                counter = counter + 1 if current_ear < config.EAR_THRESHOLD else max(0, counter - 1)
                yawn_counter = yawn_counter + 1 if current_mar > config.MAR_THRESHOLD else max(0, yawn_counter - 1)
            else:
                counter = max(0, counter - 1)
                yawn_counter = max(0, yawn_counter - 1)

            now = time.time()

            # ── Drowsiness alert (eyes) ──
            if counter >= config.CONSEC_FRAMES and now - last_alert > config.ALERT_COOLDOWN_SEC:
                alert_count += 1
                print(f"[ALERT #{alert_count}] Drowsiness detected EAR={current_ear:.3f}")
                play_alert()
                log_alert("drowsiness_detected", {"ear": round(current_ear, 3)})
                last_alert = now

            # ── Yawn alert (mouth) ──
            if yawn_counter >= config.YAWN_CONSEC_FRAMES and now - last_yawn_alert > config.YAWN_COOLDOWN_SEC:
                yawn_count += 1
                print(f"[YAWN #{yawn_count}] Yawn detected MAR={current_mar:.3f}")
                play_yawn_alert()
                log_alert("yawn_detected", {"mar": round(current_mar, 3)})
                last_yawn_alert = now

            # ── Phone-use alert ──
            phone_detected = False
            phone_confidence = 0.0
            if phone_detector:
                phone_detected, phone_confidence, phone_alert_fired = phone_detector.process(frame, now)
                if phone_alert_fired:
                    print(f"[PHONE #{phone_detector.alert_count}] Phone detected conf={phone_confidence:.3f}")
                    play_phone_alert()

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

                ear_text = f"EAR: {current_ear:.3f} ({counter}/{config.CONSEC_FRAMES})  Alerts:{alert_count}"
                mar_text = f"MAR: {current_mar:.3f} ({yawn_counter}/{config.YAWN_CONSEC_FRAMES})  Yawns:{yawn_count}"
                fps_text = f"FPS: {current_fps:.1f}"

                cv2.putText(frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE if is_drowsy else GREEN, 2)
                cv2.putText(frame, mar_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if is_yawning else GREEN, 2)
                cv2.putText(frame, fps_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

                if phone_detector:
                    phone_detector.draw(frame)
                    phone_text = f"Phone: {phone_confidence:.2f}  Alerts:{phone_detector.alert_count}"
                    cv2.putText(frame, phone_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if phone_detected else GREEN, 2)

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
    print(f"[INFO] Session ended. Drowsiness alerts: {alert_count}  Yawns: {yawn_count}{phone_summary}")
    print(f"[INFO] Per-frame log saved to {frame_log.path}")
