"""
test_sunglasses.py
-------------------
Standalone webcam test for sunglasses detection, using MediaPipe FaceLandmarker 

Combines eye-region darkness with iris/sclera color difference:
a bare eye shows a clear color contrast, a lens flattens both to one dark tone.

Press 'q' to quit.
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "face_landmarker.task")
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

RELATIVE_BRIGHTNESS_RATIO = 0.55   # eye region must be darker than this fraction of forehead brightness
SATURATION_STD_THRESHOLD = 26      # saturation spread inside eye patch below this counts as "flat tint"

LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_CONTOUR = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
FOREHEAD_IDX = 10


def polygon_mask(frame_shape, landmarks, contour_idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in contour_idx], dtype=np.int32)
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts


def eye_darkness_and_saturation(frame, landmarks, contour_idx, w, h):
    mask, pts = polygon_mask(frame.shape, landmarks, contour_idx, w, h)
    x, y, bw, bh = cv2.boundingRect(pts)
    if bw == 0 or bh == 0:
        return None, None, pts

    eye_pixels = cv2.bitwise_and(frame, frame, mask=mask)[y:y + bh, x:x + bw]
    eye_mask = mask[y:y + bh, x:x + bw] > 0

    gray = cv2.cvtColor(eye_pixels, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(eye_pixels, cv2.COLOR_BGR2HSV)

    valid_gray = gray[eye_mask]
    valid_sat = hsv[:, :, 1][eye_mask]
    if valid_gray.size == 0:
        return None, None, pts

    brightness = valid_gray.mean()
    saturation_std = valid_sat.std()

    return brightness, saturation_std, pts


def forehead_brightness(frame, landmarks, w, h, patch_radius=8):
    x, y = int(landmarks[FOREHEAD_IDX].x * w), int(landmarks[FOREHEAD_IDX].y * h)
    y1, y2 = max(0, y - patch_radius), min(h, y + patch_radius)
    x1, x2 = max(0, x - patch_radius), min(w, x + patch_radius)
    patch = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return patch.mean() if patch.size else None


def build_face_landmarker():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Missing {MODEL_PATH}")
        print(f"Download it first: wget -O {MODEL_PATH} {MODEL_DOWNLOAD_URL}")
        raise SystemExit(1)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def main():
    face_mesh = build_face_landmarker()
    cap = cv2.VideoCapture(0)
    start_time = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        result = face_mesh.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_brightness, left_sat_std, left_pts = eye_darkness_and_saturation(
                frame, landmarks, LEFT_EYE_CONTOUR, w, h
            )
            right_brightness, right_sat_std, right_pts = eye_darkness_and_saturation(
                frame, landmarks, RIGHT_EYE_CONTOUR, w, h
            )
            forehead = forehead_brightness(frame, landmarks, w, h)

            cv2.polylines(frame, [left_pts], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_pts], True, (0, 255, 0), 1)

            if left_brightness is not None and right_brightness is not None and forehead:
                left_ratio = left_brightness / forehead
                right_ratio = right_brightness / forehead

                sunglasses_on = (
                    left_ratio < RELATIVE_BRIGHTNESS_RATIO and left_sat_std < SATURATION_STD_THRESHOLD
                    and right_ratio < RELATIVE_BRIGHTNESS_RATIO and right_sat_std < SATURATION_STD_THRESHOLD
                )

                status = "SUNGLASSES ON" if sunglasses_on else "BARE EYES"
                color = (0, 0, 255) if sunglasses_on else (0, 255, 0)
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                debug_text = f"L: ratio={left_ratio:.2f} satstd={left_sat_std:.0f}  R: ratio={right_ratio:.2f} satstd={right_sat_std:.0f}  forehead={forehead:.0f}"
                cv2.putText(frame, debug_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                # cv2.putText(frame, debug_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                print(debug_text)
        else:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Sunglasses Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()