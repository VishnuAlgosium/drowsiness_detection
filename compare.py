"""
compare_cigarette_models.py
----------------------------
Runs the same face crops through the cigarette classifier's .pt weights and
its NCNN export, side by side, so any accuracy drop from the ncnn conversion
shows up as either a class-name mismatch or a per-image confidence gap.

Usage:
    python compare_cigarette_models.py \
        --images-dir ./test_images \
        --pt-model ./models/cigarette_classification_v1.pt \
        --ncnn-model ./models/cigarette_classification_v1_ncnn_model \
        --face-model ./models/face_landmarker.task \
        --padding 0.3
"""

import argparse
import glob
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


def build_face_landmarker(model_path: str):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def padded_face_box(landmarks, w: int, h: int, padding_ratio: float):
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


def get_face_crop(face_landmarker, image, padding: float):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_landmarker.detect(mp_image)
    if not results.face_landmarks:
        return None
    x1, y1, x2, y2 = padded_face_box(results.face_landmarks[0], w, h, padding)
    return image[y1:y2, x1:x2]


def top_prediction(model, crop, img_size: int):
    result = model.predict(source=crop, imgsz=img_size, verbose=False)[0]
    probs = result.probs
    top1_idx = int(probs.top1)
    return result.names[top1_idx], float(probs.data[top1_idx])


def main():
    parser = argparse.ArgumentParser(description="Compare .pt vs NCNN cigarette classifier outputs.")
    parser.add_argument("--images-dir", required=True, help="Folder of test images (jpg/png)")
    parser.add_argument("--pt-model", required=True, help="Path to .pt classification weights")
    parser.add_argument("--ncnn-model", required=True, help="Path to the *_ncnn_model export folder")
    parser.add_argument("--face-model", default="./models/face_landmarker.task")
    parser.add_argument("--padding", type=float, default=0.3)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    print("[INFO] Loading models...")
    face_landmarker = build_face_landmarker(args.face_model)
    pt_model = YOLO(args.pt_model, task="classify")
    ncnn_model = YOLO(args.ncnn_model, task="classify")

    print(f"[INFO] PT model classes:   {pt_model.names}")
    print(f"[INFO] NCNN model classes: {ncnn_model.names}")
    if pt_model.names != ncnn_model.names:
        print("[WARN] Class name/index mapping differs between the two exports!\n")

    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(args.images_dir, ext))
    )
    if not image_paths:
        print(f"[FAIL] No images found in {args.images_dir}")
        return

    print(f"\n[INFO] Comparing {len(image_paths)} images\n")
    print(f"{'image':<30} {'pt label':<12} {'pt conf':<9} {'ncnn label':<12} {'ncnn conf':<10} {'diff':<7} agree")
    print("-" * 100)

    mismatches = 0
    conf_diffs = []

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Could not read {path}, skipping")
            continue

        crop = get_face_crop(face_landmarker, image, args.padding)
        if crop is None or crop.size == 0:
            print(f"{os.path.basename(path):<30} no face detected, skipping")
            continue

        pt_label, pt_conf = top_prediction(pt_model, crop, args.img_size)
        ncnn_label, ncnn_conf = top_prediction(ncnn_model, crop, args.img_size)

        agree = pt_label == ncnn_label
        mismatches += 0 if agree else 1
        conf_diffs.append(abs(pt_conf - ncnn_conf))

        print(
            f"{os.path.basename(path):<30} {pt_label:<12} {pt_conf:<9.3f} "
            f"{ncnn_label:<12} {ncnn_conf:<10.3f} {abs(pt_conf - ncnn_conf):<7.3f} "
            f"{'YES' if agree else 'NO -- MISMATCH'}"
        )

    if conf_diffs:
        print("\n[SUMMARY]")
        print(f"  Label mismatches:   {mismatches}/{len(conf_diffs)}")
        print(f"  Avg confidence diff: {sum(conf_diffs) / len(conf_diffs):.3f}")
        print(f"  Max confidence diff: {max(conf_diffs):.3f}")


if __name__ == "__main__":
    main()