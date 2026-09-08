"""
Standalone script: show ALL MediaPipe FaceLandmarker (Tasks API) keypoints,
each labeled with its index number, on a live webcam feed.

Written against mediapipe==1.0.0's Tasks API (FaceLandmarker), matching the
same API your drowsy_detect pipeline is built on -- NOT the older
mp.solutions.face_mesh legacy API, which may not be available/reliable in
this mediapipe version.

Requirements:
    pip install mediapipe opencv-python numpy --break-system-packages

You need a face_landmarker.task model file. You already have one at, e.g.:
    /home/megha/Downloads/face_landmarker.task
    /home/megha/Documents/alGO/drowsiness_detection/drowsiness_detection/models/face_landmarker.task

Update MODEL_PATH below to point to it.

Controls:
    q       - quit
    n       - toggle landmark index numbers on/off
    +/-     - increase/decrease dot size
    z / x   - zoom in / out (crop + resize around face center)
    s       - save current frame to disk (annotated_frame_<timestamp>.png)
"""

import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MODEL_PATH = "/home/megha/Downloads/face_landmarker.task"  # <-- update if needed

CAMERA_INDEX = 2
FRAME_WIDTH = 900

DOT_RADIUS_DEFAULT = 1
FONT_SCALE = 0.28
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 255)   # yellow-ish (BGR)
DOT_COLOR = (0, 0, 255)      # red


def build_landmarker(model_path: str) -> mp_vision.FaceLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def main():
    landmarker = build_landmarker(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    show_numbers = True
    dot_radius = DOT_RADIUS_DEFAULT
    zoom = 1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] failed to read frame from camera")
            break

        new_h = int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])
        frame = cv2.resize(frame, (FRAME_WIDTH, new_h))
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]  # first detected face

            for idx, lm in enumerate(landmarks):
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), dot_radius, DOT_COLOR, -1)
                if show_numbers:
                    cv2.putText(
                        frame, str(idx), (x + 2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                        TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA,
                    )

            cv2.putText(
                frame, f"landmarks: {len(landmarks)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            if zoom > 1.0:
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
                half_w = int(w / (2 * zoom))
                half_h = int(h / (2 * zoom))
                x0 = max(0, cx - half_w)
                x1 = min(w, cx + half_w)
                y0 = max(0, cy - half_h)
                y1 = min(h, cy + half_h)
                cropped = frame[y0:y1, x0:x1]
                if cropped.size > 0:
                    frame = cv2.resize(cropped, (w, h))
        else:
            cv2.putText(
                frame, "no face detected", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        cv2.putText(
            frame,
            "q:quit  n:numbers  +/-:dot size  z/x:zoom  s:save",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        cv2.imshow("Face Landmarker - All Keypoints", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            show_numbers = not show_numbers
        elif key in (ord("+"), ord("=")):
            dot_radius = min(6, dot_radius + 1)
        elif key == ord("-"):
            dot_radius = max(1, dot_radius - 1)
        elif key == ord("z"):
            zoom = min(4.0, zoom + 0.25)
        elif key == ord("x"):
            zoom = max(1.0, zoom - 0.25)
        elif key == ord("s"):
            fname = f"annotated_frame_{int(time.time()*1000)}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()