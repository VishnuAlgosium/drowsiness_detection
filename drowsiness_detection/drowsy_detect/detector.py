"""
detector.py
-----------
Ties together camera capture, MediaPipe FaceLandmarker, EAR scoring,
display, keyboard control, and audio alerting into the main run loop.
"""

import os
import sys
import time

import cv2

from . import config
from .audio import play_alert, shutdown_audio, AUDIO_AVAILABLE
from .ear import average_ear
from .display import LazyDisplay
from .keyboard_input import KeyReader


def _check_model() -> None:
    if not os.path.exists(config.MODEL_PATH):
        print(f"[ERROR] Missing {config.MODEL_PATH}")
        print("Download it first:")
        print(f"wget -O {config.MODEL_PATH} {config.MODEL_DOWNLOAD_URL}")
        sys.exit(1)


def _build_face_landmarker():
    # Imported here (after config.apply_cpu_settings()) so env vars take effect first.
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

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)

    display = LazyDisplay(config.DISPLAY_W, config.DISPLAY_H, config.CAM_FPS)
    display_on = config.DISPLAY_ON_START

    if display_on:
        display.start()

    keys = KeyReader()

    counter = 0
    alert_count = 0
    last_alert = 0.0

    print()
    print("[INFO] Running")
    print("[INFO] Keys: v = toggle display, q = quit")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Camera frame failed")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            results = face_mesh.detect_for_video(mp_image, timestamp_ms)

            current_ear = 0.0

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                current_ear = average_ear(
                    landmarks, config.LEFT_EYE_IDX, config.RIGHT_EYE_IDX, w, h
                )

                if current_ear < config.EAR_THRESHOLD:
                    counter += 1
                else:
                    counter = max(0, counter - 1)
            else:
                counter = max(0, counter - 1)

            # ── Drowsiness alert ──
            now = time.time()

            if counter >= config.CONSEC_FRAMES:
                if now - last_alert > config.ALERT_COOLDOWN_SEC:
                    alert_count += 1
                    print(f"[ALERT #{alert_count}] Drowsiness detected EAR={current_ear:.3f}")
                    play_alert()
                    last_alert = now

            # ── Display ──
            if display_on:
                text = (
                    f"EAR: {current_ear:.3f} "
                    f"Counter: {counter}/{config.CONSEC_FRAMES} "
                    f"Alerts:{alert_count}"
                )

                color = (0, 0, 255) if counter >= config.CONSEC_FRAMES else (0, 255, 0)

                cv2.putText(
                    frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

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

    print(f"[INFO] Session ended. Alerts: {alert_count}")
