"""
crop_face.py
------------
Crops the face out of an image using MediaPipe FaceLandmarker, with some
padding around it so the crop isn't a tight, awkward box.

Usage:
    python crop_face.py --image photo.jpg --model face_landmarker.task --output face_crop.jpg
"""

import argparse
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def build_face_landmarker(model_path: str):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_padded_face_box(landmarks, w: int, h: int, padding_ratio: float):
    """
    Bounding box around all face landmarks, expanded by padding_ratio of the
    face's own size on each side, then clamped to stay inside the image.
    """
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser(description="Crop a face out of an image, with padding.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to face_landmarker.task")
    parser.add_argument("--output", default="face_crop.jpg", help="Path to save the cropped face")
    parser.add_argument("--padding", type=float, default=0.3, help="Padding as a ratio of face size (default: 0.3)")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[FAIL] Could not read image: {args.image}")
        sys.exit(1)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    face_landmarker = build_face_landmarker(args.model)
    results = face_landmarker.detect(mp_image)

    if not results.face_landmarks:
        print("[FAIL] No face detected")
        sys.exit(1)

    landmarks = results.face_landmarks[0]
    x1, y1, x2, y2 = get_padded_face_box(landmarks, w, h, args.padding)

    face_crop = frame[y1:y2, x1:x2]
    cv2.imwrite(args.output, face_crop)
    print(f"[INFO] Saved cropped face to {args.output} ({x2 - x1}x{y2 - y1})")


if __name__ == "__main__":
    main()