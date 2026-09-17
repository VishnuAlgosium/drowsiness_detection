"""
test_smoking_pipeline.py
-------------------------
Standalone end-to-end test of the actual fused smoking-detection pipeline
described in smoking.py/detector.py -- not just the object detector in
isolation (that's what validate_yolo_model.py checks).

This runs, frame by frame, on a video file or webcam:
  1. Face landmarks  -> mouth position
  2. Hand landmarks   -> hand-to-mouth distance ratio
  3. Cycle monitor     -> counts hand-to-mouth approach cycles over time
  4. Cigarette model   -> runs ONLY while a hand is near the mouth
  5. Fusion            -> alert only when both the object is confirmed
                          AND enough cycles happened recently

Use this to sanity-check the real decision path (does the hand landmark
track reliably, does the cycle count behave on an actual smoking video,
does the cigarette model actually fire when it should) before this logic
is embedded in the full DMS run loop.

Usage:
    python test_smoking_pipeline.py --source path/to/video.mp4 \
        --face-model face_landmarker.task --hand-model hand_landmarker.task \
        --cigarette-model models/smoking_detection_ncnn_model --display

    python test_smoking_pipeline.py --source 0 \
        --face-model face_landmarker.task --hand-model hand_landmarker.task \
        --cigarette-model models/smoking_detection_ncnn_model --display
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import config

# ─────────────────────────────────────────────
# Landmark indices (MediaPipe FaceMesh / Hands topology)
# ─────────────────────────────────────────────
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14
LEFT_EYE_OUTER_IDX = 362
RIGHT_EYE_OUTER_IDX = 33
HAND_THUMB_TIP_IDX = 4
HAND_INDEX_TIP_IDX = 8


class MonotonicTimestamp:
    """Strictly increasing ms timestamps, required by MediaPipe's VIDEO mode."""

    def __init__(self):
        self._last_ms = -1

    def next(self) -> int:
        ms = int(time.monotonic() * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        return ms


def hand_to_mouth_ratio(hand_landmarks, mouth_x: float, mouth_y: float, w: int, h: int, scale: float) -> float:
    """Distance from a hand's pinch point (thumb+index tip midpoint) to the
    mouth, as a ratio of `scale` (interocular distance) so it's independent
    of how close the person is to the camera. Lower = nearer the mouth."""
    thumb = hand_landmarks[HAND_THUMB_TIP_IDX]
    index = hand_landmarks[HAND_INDEX_TIP_IDX]

    pinch_x = (thumb.x + index.x) / 2.0 * w
    pinch_y = (thumb.y + index.y) / 2.0 * h

    distance = ((pinch_x - mouth_x) ** 2 + (pinch_y - mouth_y) ** 2) ** 0.5
    return distance / scale if scale > 0 else float("inf")


class HandMouthCycleMonitor:
    """Counts hand-near-mouth rising edges (cycles) in a trailing window."""

    def __init__(self, window_sec: float):
        self.window_sec = window_sec
        self._history = deque()  # (timestamp, hand_near_mouth) pairs

    def update(self, now: float, hand_near_mouth: bool) -> int:
        self._history.append((now, hand_near_mouth))

        while self._history and now - self._history[0][0] > self.window_sec:
            self._history.popleft()

        return sum(
            1 for i in range(1, len(self._history))
            if self._history[i][1] and not self._history[i - 1][1]
        )


class SmokingDetector:
    """Cigarette object detection, gated to only run while a hand is near
    the mouth. Reports whether the model has consistently confirmed a
    cigarette over the last few detection attempts."""

    def __init__(self, model_path: str, class_name: str, img_size: int, conf_threshold: float,
                 detect_every_n_frames: int, confirm_frames: int, confirm_window: int):
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.detect_every_n_frames = detect_every_n_frames
        self.confirm_frames = confirm_frames

        self.class_idx = next(
            (idx for idx, name in self.model.names.items() if name == class_name), None
        )
        if self.class_idx is None:
            print(f"[FAIL] Class '{class_name}' not found. Available: {list(self.model.names.values())}")
            sys.exit(1)

        self.confirm_buffer = deque(maxlen=confirm_window)
        self.frame_num = 0
        self.last_results = None

    def process(self, frame, hand_near_mouth: bool):
        self.frame_num += 1
        should_run = hand_near_mouth and (self.frame_num % self.detect_every_n_frames == 0)

        if should_run:
            self.last_results = self.model.predict(
                source=frame, imgsz=self.img_size, verbose=False, device="cpu"
            )

        confidence = self._max_confidence() if hand_near_mouth else 0.0
        detected = confidence >= self.conf_threshold

        self.confirm_buffer.append(detected)
        confirmed = (
            len(self.confirm_buffer) == self.confirm_buffer.maxlen
            and sum(self.confirm_buffer) >= self.confirm_frames
        )
        return detected, confidence, confirmed

    def draw(self, frame) -> None:
        if not self.last_results:
            return
        for box in self.last_results[0].boxes:
            if int(box.cls.item()) != self.class_idx:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)

    def _max_confidence(self) -> float:
        if not self.last_results:
            return 0.0
        confidences = [
            float(b.conf.item()) for b in self.last_results[0].boxes
            if int(b.cls.item()) == self.class_idx
        ]
        return max(confidences) if confidences else 0.0


def build_face_landmarker(model_path: str):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options), mp


def build_hand_landmarker(model_path: str, mp, max_hands: int = 2):
    if not os.path.exists(model_path):
            print(f"[ERROR] Missing {model_path}")
            print("Download it first:")
            print(f"wget -O {model_path} https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            sys.exit(1)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser(description="Test the fused smoking-detection pipeline end-to-end.")
    parser.add_argument("--source", required=False, help="Video file path, or '0' for the default webcam", default="0")
    parser.add_argument("--face-model", required=False, help="Path to face_landmarker.task", default="../models/face_landmarker.task")
    parser.add_argument("--hand-model", required=False, help="Path to hand_landmarker.task", default="../models/hand_landmarker.task")
    parser.add_argument("--cigarette-model", required=False, help="Path to cigarette NCNN model folder", default="../models/cigarette_detection_v3_ncnn_model")
    parser.add_argument("--class-name", default="cigarette")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-threshold", type=float, default=0.45)
    parser.add_argument("--near-mouth-ratio", type=float, default=1.1,
                         help="Hand counts as 'near mouth' below this ratio of interocular distance")
    parser.add_argument("--detect-every-n-frames", type=int, default=3)
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--confirm-window", type=int, default=5)
    parser.add_argument("--cycle-window-sec", type=float, default=45.0)
    parser.add_argument("--min-cycles", type=int, default=2)
    parser.add_argument("--cooldown-sec", type=float, default=20.0)
    parser.add_argument("--display", action="store_true", help="Show a live annotated window", default=True)
    args = parser.parse_args()

    for path, label in [(args.face_model, "face model"), (args.hand_model, "hand model")]:
        if not os.path.exists(path):
            print(f"[FAIL] Missing {label}: {path}")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[FAIL] Could not open source: {args.source}")
        sys.exit(1)

    print("[INFO] Loading face landmarker...")
    face_mesh, mp = build_face_landmarker(args.face_model)
    print("[INFO] Loading hand landmarker...")
    hand_landmarker = build_hand_landmarker(args.hand_model, mp)
    print("[INFO] Loading cigarette model...")
    smoking_detector = SmokingDetector(
        args.cigarette_model, args.class_name, args.img_size, args.conf_threshold,
        args.detect_every_n_frames, args.confirm_frames, args.confirm_window,
    )
    cycle_monitor = HandMouthCycleMonitor(args.cycle_window_sec)
    timestamps = MonotonicTimestamp()

    frame_num = 0
    alert_count = 0
    last_alert_time = 0.0
    print("[INFO] Running. Press 'q' in the display window to quit (if --display).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream")
                break

            frame_num += 1
            now = time.time()
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = timestamps.next()
            face_results = face_mesh.detect_for_video(mp_image, ts)
            hand_results = hand_landmarker.detect_for_video(mp_image, ts)

            hand_near_mouth = False
            closest_ratio = None

            if face_results.face_landmarks and hand_results.hand_landmarks:
                landmarks = face_results.face_landmarks[0]
                mouth_x = (landmarks[MOUTH_TOP_IDX].x + landmarks[MOUTH_BOTTOM_IDX].x) / 2.0 * w
                mouth_y = (landmarks[MOUTH_TOP_IDX].y + landmarks[MOUTH_BOTTOM_IDX].y) / 2.0 * h

                eye_l = landmarks[LEFT_EYE_OUTER_IDX]
                eye_r = landmarks[RIGHT_EYE_OUTER_IDX]
                interocular = (((eye_l.x - eye_r.x) * w) ** 2 + ((eye_l.y - eye_r.y) * h) ** 2) ** 0.5

                closest_ratio = min(
                    hand_to_mouth_ratio(hand, mouth_x, mouth_y, w, h, interocular)
                    for hand in hand_results.hand_landmarks
                )
                hand_near_mouth = closest_ratio < args.near_mouth_ratio

            cycle_count = cycle_monitor.update(now, hand_near_mouth)
            detected, confidence, object_confirmed = smoking_detector.process(frame, hand_near_mouth)
            smoking_confirmed = object_confirmed and cycle_count >= args.min_cycles

            alert_fired = False
            if smoking_confirmed and now - last_alert_time > args.cooldown_sec:
                alert_count += 1
                alert_fired = True
                last_alert_time = now

            ratio_str = f"{closest_ratio:.2f}" if closest_ratio is not None else "n/a"
            status = (
                f"[{frame_num:5d}] hand_near_mouth={hand_near_mouth!s:5} ratio={ratio_str:>5} "
                f"cig_conf={confidence:.2f} object_confirmed={object_confirmed!s:5} "
                f"cycles={cycle_count}/{args.min_cycles}"
            )
            if alert_fired:
                status += f"  <<< SMOKING ALERT #{alert_count} >>>"
            print(status)

            if args.display:
                smoking_detector.draw(frame)
                color = (0, 0, 255) if smoking_confirmed else (0, 255, 0)
                cv2.putText(frame, f"near_mouth:{hand_near_mouth} cig:{confidence:.2f} cycles:{cycle_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("smoking pipeline test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"[INFO] Done. Frames processed: {frame_num}  Smoking alerts: {alert_count}")


if __name__ == "__main__":
    main()