#!/usr/bin/env python3
"""
test_blink.py
-------------
Standalone check that blink detection (EAR dropping below threshold and
recovering) works before running the full detector. Prints a live EAR
reading and a running blink count; press 'q' to quit.

Usage:
    python test_blink.py
"""

import sys
import time

import cv2

import config
from ear import average_ear

BLINK_EAR_THRESHOLD = 0.25   # EAR below this counts as "eyes closed"
BLINK_MIN_CLOSED_FRAMES = 2  # avoid counting single-frame landmark noise as a blink


def build_face_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options), mp


def main() -> None:
    face_mesh, mp = build_face_landmarker()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    blink_count = 0
    closed_frames = 0
    eyes_were_closed = False
    start_time_ms = -1

    print("[INFO] Running blink test. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= start_time_ms:
                timestamp_ms = start_time_ms + 1
            start_time_ms = timestamp_ms

            result = face_mesh.detect_for_video(mp_image, timestamp_ms)

            ear = 0.0
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                ear = average_ear(landmarks, config.LEFT_EYE_IDX, config.RIGHT_EYE_IDX, w, h)

                eyes_closed = ear < BLINK_EAR_THRESHOLD
                if eyes_closed:
                    closed_frames += 1
                    if closed_frames >= BLINK_MIN_CLOSED_FRAMES:
                        eyes_were_closed = True
                else:
                    if eyes_were_closed:
                        blink_count += 1
                        print(f"[BLINK #{blink_count}] EAR={ear:.3f}")
                    closed_frames = 0
                    eyes_were_closed = False

            cv2.putText(frame, f"EAR: {ear:.3f}  Blinks: {blink_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Blink Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Total blinks detected: {blink_count}")


if __name__ == "__main__":
    main()