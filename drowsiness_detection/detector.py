"""Main drowsiness detection loop: capture, MediaPipe inference, alerting, display."""

import sys
import time
from collections import deque

import cv2
cv2.setNumThreads(4)  
cv2.ocl.setUseOpenCL(False)  # Disable OpenCL to avoid GPU conflicts with ffplay
import mediapipe as mp
import numpy as np

from . import config as cfg
from .audio import AUDIO_AVAILABLE, play_alert_sound, shutdown_audio
from .display import FFplayDisplay
from .drawing import draw_eye_contour, draw_iris_dots, draw_hud, draw_alert_overlay, draw_no_face
from .ear import eye_aspect_ratio
from .input_keys import TerminalKeyReader
import os
from datetime import datetime


def run():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check that camera index 0 is available.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, cfg.CAM_FPS)

    ear_thresh   = cfg.EAR_THRESHOLD
    frame_thresh = cfg.CONSEC_FRAMES
    save_dir     = cfg.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    counter      = 0
    alert_count  = 0
    last_alert   = 0.0
    alert_active = False

    fps_history  = deque(maxlen=30)
    prev_time    = time.time()

    flash_state  = True
    flash_timer  = 0.0

    print("\n" + "─" * 50)
    print("  Driver Drowsiness Detection — Running (ffplay/GPU display)")
    print("─" * 50)
    print("  Q        quit")
    print("  + / -    adjust EAR threshold")
    print("  [ / ]    adjust frame threshold")
    print("  R        reset counters")
    print("  (keys are read from THIS terminal, not the video window)")
    if not AUDIO_AVAILABLE:
        print("  [!] Audio unavailable — install pygame for sound alerts")
    print("─" * 50 + "\n")

    display = FFplayDisplay(cfg.DISPLAY_W, cfg.DISPLAY_H, title="Drowsiness Detection", fps=30)
    keys = TerminalKeyReader()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()   # Original image with no drawings
            h, w  = frame.shape[:2]

            now      = time.time()
            dt       = now - prev_time
            prev_time = now
            fps_history.append(1.0 / dt if dt > 0 else 0)
            fps = np.mean(fps_history)

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            current_ear = 0.0
            eye_closed  = False

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                ear_l = eye_aspect_ratio(lm, cfg.LEFT_EYE_IDX,  w, h)
                ear_r = eye_aspect_ratio(lm, cfg.RIGHT_EYE_IDX, w, h)
                current_ear = (ear_l + ear_r) / 2.0
                eye_closed  = current_ear < ear_thresh

                if eye_closed:
                    counter += 1
                else:
                    counter = max(0, counter - 1)

                draw_eye_contour(frame, lm, cfg.LEFT_EYE_IDX,  w, h, eye_closed)
                draw_eye_contour(frame, lm, cfg.RIGHT_EYE_IDX, w, h, eye_closed)
                draw_iris_dots(frame, lm, cfg.LEFT_IRIS_IDX,  w, h)
                draw_iris_dots(frame, lm, cfg.RIGHT_IRIS_IDX, w, h)
            else:
                draw_no_face(frame)
                counter = max(0, counter - 2)

            alert_level = None
            if counter >= frame_thresh:
                alert_level = "danger"
                if not alert_active:
                    alert_active = True
                    alert_count += 1

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_path = os.path.join(
                        save_dir,
                        f"drowsy_{alert_count}_{timestamp}.jpg"
                    )

                    cv2.imwrite(image_path, raw_frame)

                    print(
                        f"[ALERT #{alert_count}] Drowsiness detected at "
                        f"{time.strftime('%H:%M:%S')} | Saved: {image_path}"
                    )
                if now - last_alert > cfg.ALERT_COOLDOWN_SEC:
                    play_alert_sound()
                    last_alert = now
            elif counter >= frame_thresh // 2:
                alert_level = "warn"
                alert_active = False
            else:
                alert_active = False

            if alert_level == "danger":
                if now - flash_timer > 0.4:
                    flash_state = not flash_state
                    flash_timer = now
                if flash_state:
                    draw_alert_overlay(frame, "danger")
            elif alert_level == "warn":
                draw_alert_overlay(frame, "warn")

            draw_hud(frame, {
                "ear":          current_ear,
                "ear_thresh":   ear_thresh,
                "counter":      counter,
                "frame_thresh": frame_thresh,
                "fps":          fps,
                "alert_count":  alert_count,
            })

            if not display.show(frame):
                print("\nffplay window closed. Exiting...")
                break

            key = keys.get_key()
            if key:
                key = key.lower()
                if key == 'q':
                    print("\nExiting...")
                    break
                elif key in ('+', '='):
                    ear_thresh = min(0.40, round(ear_thresh + 0.01, 2))
                    print(f"  EAR threshold → {ear_thresh:.2f}")
                elif key == '-':
                    ear_thresh = max(0.10, round(ear_thresh - 0.01, 2))
                    print(f"  EAR threshold → {ear_thresh:.2f}")
                elif key == ']':
                    frame_thresh = min(120, frame_thresh + 4)
                    print(f"  Frame threshold → {frame_thresh}")
                elif key == '[':
                    frame_thresh = max(8, frame_thresh - 4)
                    print(f"  Frame threshold → {frame_thresh}")
                elif key == 'r':
                    counter = 0
                    alert_count = 0
                    alert_active = False
                    print("  Counters reset.")

    finally:
        keys.restore()
        display.close()
        face_mesh.close()
        cap.release()
        shutdown_audio()

    print(f"\nSession ended. Total drowsiness alerts: {alert_count}")
