"""
emotion_test_media.py
----------------------
Test the blendshape-based "emotion" heuristic against static images,
video FILES, and live webcam/RTSP streams.

This exists to debug the classifier: run it against known, labeled
expressions and inspect the raw blendshape scores + predicted label.

Usage:
    # Single image
    python emotion_test_media.py --image path/to/smile.jpg

    # Folder of images (e.g. one folder per expression for eyeballing)
    python emotion_test_media.py --image-dir path/to/test_images/

    # Video file
    python emotion_test_media.py --video path/to/clip.mp4

    # Live webcam (device index) or RTSP stream, with FFplay preview
    python emotion_test_media.py --webcam 0
    python emotion_test_media.py --webcam rtsp://user:pass@host/stream --no-display

    # Any of the above + write annotated output (webcam/RTSP writes a video file)
    python emotion_test_media.py --image path/to/smile.jpg --out-dir results/

All runs also write a CSV log (default: emotion_log.csv) with every
blendshape score per frame/image, so you can plot/inspect the raw
numbers instead of just trusting the predicted label.

Yawn detection: video/webcam runs also track a sustained-jawOpen
"yawning" flag (separate from the happy/sad/... emotion label) via
--yawn-threshold and --yawn-min-duration. Single images/frames only
get an unconfirmed "possible yawn" signal, since duration can't be
measured from one frame.

Requires the same face_landmarker.task model as emotion_check.py.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL_PATH = "drowsiness_detection/models/face_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# All 52 ARKit blendshape names, in a fixed order, so the CSV header is
# stable across runs even if a given frame is missing some categories.
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]


def _check_model(model_path: str) -> None:
    if not os.path.exists(model_path):
        print(f"[ERROR] Missing {model_path}")
        print("Download it first:")
        print(f"wget -O {model_path} {MODEL_DOWNLOAD_URL}")
        sys.exit(1)


def _build_face_landmarker(model_path: str, video_mode: bool):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.FaceLandmarker.create_from_options(options), mp


def _blendshape_dict(face_results) -> dict:
    if not face_results.face_blendshapes:
        return {}
    return {b.category_name: b.score for b in face_results.face_blendshapes[0]}


def _estimate_emotion(scores: dict) -> tuple[str, float]:
    """
    Same heuristic as emotion_check.py, EXCEPT the "neutral" term is fixed.

    The original formula `1.0 - max(other_scores)` gives neutral a
    structural head start (e.g. a 0.4 smile score still loses to a 0.6
    neutral score), which silently swallows real expressions. This
    version scores neutral the same way as the others: how *un*-active
    all the expression channels are, scaled so it's comparable rather
    than dominant. Tune the 0.5 baseline/weight to your own data once
    you've logged some real distributions (see CSV output).
    """
    if not scores:
        return "no_face", 0.0

    smile = max(scores.get("mouthSmileLeft", 0.0), scores.get("mouthSmileRight", 0.0))
    frown = max(scores.get("mouthFrownLeft", 0.0), scores.get("mouthFrownRight", 0.0))
    brow_down = max(scores.get("browDownLeft", 0.0), scores.get("browDownRight", 0.0))
    brow_up = max(scores.get("browInnerUp", 0.0), scores.get("browOuterUpLeft", 0.0), scores.get("browOuterUpRight", 0.0))
    jaw_open = scores.get("jawOpen", 0.0)
    eye_wide = max(scores.get("eyeWideLeft", 0.0), scores.get("eyeWideRight", 0.0))
    eye_squint = max(scores.get("eyeSquintLeft", 0.0), scores.get("eyeSquintRight", 0.0))
    nose_sneer = max(scores.get("noseSneerLeft", 0.0), scores.get("noseSneerRight", 0.0))

    activity = max(smile, frown, brow_down, brow_up, jaw_open, nose_sneer)

    candidates = {
        "happy": smile * (1.0 + 0.3 * eye_squint),
        "surprised": (brow_up + eye_wide + jaw_open) / 3.0,
        "angry": (brow_down + nose_sneer) / 2.0,
        "sad": (frown + brow_down * 0.3),
        "neutral": max(0.0, 0.5 - activity),
    }

    label = max(candidates, key=candidates.get)
    return label, candidates[label]


class CsvLogger:
    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["source", "frame_or_time", "label", "confidence", "yawning"] + BLENDSHAPE_NAMES
        )

    def log(self, source: str, frame_or_time, label: str, confidence: float, scores: dict, yawning: bool = False):
        row = [source, frame_or_time, label, f"{confidence:.4f}", str(yawning)]
        row += [f"{scores.get(name, 0.0):.4f}" for name in BLENDSHAPE_NAMES]
        self._writer.writerow(row)

    def close(self):
        self._file.close()


class YawnDetector:
    """Flags a yawn from a SUSTAINED high jawOpen score, not a single frame.

    A single wide-open-mouth frame is just as likely to be talking,
    laughing, or a bad landmark fit. Requiring `jawOpen` to stay above
    `open_threshold` for at least `min_duration` seconds filters most
    of that out. `cooldown` prevents one long yawn from re-triggering
    repeatedly right after it ends.

    Stateful and per-stream: create one instance per video/webcam run,
    call `update()` once per frame with the current jawOpen score and
    a monotonically increasing timestamp (seconds).
    """

    def __init__(self, open_threshold: float = 0.5, min_duration: float = 0.4, cooldown: float = 1.0):
        self.open_threshold = open_threshold
        self.min_duration = min_duration
        self.cooldown = cooldown
        self._open_since: float | None = None
        self._last_yawn_end: float | None = None
        self.is_yawning = False

    def update(self, jaw_open_score: float, timestamp_sec: float) -> bool:
        """Returns True if the current frame is part of an active yawn."""
        print(f"[DEBUG] jawOpen={jaw_open_score:.3f}, timestamp={timestamp_sec:.2f}, ")
        if jaw_open_score >= self.open_threshold:
            if self._open_since is None:
                self._open_since = timestamp_sec
            open_duration = timestamp_sec - self._open_since
            self.is_yawning = open_duration >= self.min_duration
            if self.is_yawning:
                self._last_yawn_end = timestamp_sec
        else:
            self._open_since = None
            self.is_yawning = False
        return self.is_yawning


class LazyDisplay:
    """Lazily starts an ffplay process and streams raw frames to it.

    Used instead of cv2.imshow, which is unreliable/unavailable in
    headless-ish or GUI-less OpenCV builds. Requires ffplay
    (`sudo apt install ffmpeg`) on PATH.
    """

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = None

    def start(self) -> None:
        if self.proc:
            return
        if shutil.which("ffplay") is None:
            print("[WARN] ffplay missing. Install: sudo apt install ffmpeg")
            return
        cmd = [
            "ffplay",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-framedrop",
            "-an",
            "-i", "-",
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def show(self, frame: np.ndarray) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(frame.tobytes())
        except Exception:
            self.proc = None

    def stop(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None

    def is_alive(self) -> bool:
        """False once the user has closed the ffplay window."""
        return self.proc is not None and self.proc.poll() is None


def _annotate(frame: np.ndarray, label: str, confidence: float, scores: dict, yawning: bool = False) -> np.ndarray:
    out = frame.copy()
    color = (0, 255, 0) if label in ("happy", "neutral") else (0, 165, 255)
    cv2.putText(out, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if yawning:
        cv2.putText(out, "YAWNING", (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show top-5 raw blendshapes underneath so you can eyeball the
    # numbers that drove the decision, without opening the CSV.
    if scores:
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        for i, (name, val) in enumerate(top):
            y = 55 + i * 22
            cv2.putText(out, f"{name}: {val:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def process_image(
    landmarker, mp, image_path: str, out_dir: str | None, logger: CsvLogger,
    yawn_threshold: float = 0.5,
) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[WARN] Could not read image: {image_path}")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    scores = _blendshape_dict(result)
    label, confidence = _estimate_emotion(scores)

    # A single frame can't confirm "sustained" jaw-open, so this is a
    # coarser signal than the video/webcam YawnDetector: just "mouth is
    # wide open in this frame," which could also be talking/laughing.
    jaw_open = scores.get("jawOpen", 0.0)
    possible_yawn = jaw_open >= yawn_threshold

    name = os.path.basename(image_path)
    yawn_note = " [possible yawn - single frame, unconfirmed]" if possible_yawn else ""
    print(f"[{name}] -> {label} ({confidence:.2f}){yawn_note}")
    logger.log(name, "-", label, confidence, scores, yawning=possible_yawn)

    if out_dir:
        annotated = _annotate(frame, label, confidence, scores, yawning=possible_yawn)
        out_path = os.path.join(out_dir, f"annotated_{name}")
        cv2.imwrite(out_path, annotated)
        print(f"    saved -> {out_path}")


def process_video(
    landmarker, mp, video_path: str, out_dir: str | None, logger: CsvLogger,
    yawn_threshold: float = 0.5, yawn_min_duration: float = 0.4,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = os.path.basename(video_path)

    writer = None
    if out_dir:
        out_path = os.path.join(out_dir, f"annotated_{name}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    yawn_detector = YawnDetector(open_threshold=yawn_threshold, min_duration=yawn_min_duration)
    yawn_events = 0
    was_yawning = False

    frame_idx = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires monotonically increasing timestamps in ms.
        timestamp_ms = int((frame_idx / fps) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        scores = _blendshape_dict(result)
        label, confidence = _estimate_emotion(scores)

        t_sec = frame_idx / fps
        yawning = yawn_detector.update(scores.get("jawOpen", 0.0), t_sec)
        if yawning and not was_yawning:
            yawn_events += 1
            print(f"[{name}] yawn #{yawn_events} started at {t_sec:.1f}s")
        was_yawning = yawning

        logger.log(name, f"{t_sec:.2f}", label, confidence, scores, yawning=yawning)

        if writer:
            annotated = _annotate(frame, label, confidence, scores, yawning=yawning)
            writer.write(annotated)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[{name}] frame {frame_idx} ({t_sec:.1f}s) -> {label} ({confidence:.2f})")

    cap.release()
    if writer:
        writer.release()
        print(f"    saved -> {out_path}")
    elapsed = time.time() - start
    print(f"[{name}] done: {frame_idx} frames in {elapsed:.1f}s, {yawn_events} yawn(s) detected")


def process_webcam(
    landmarker,
    mp,
    source: str,
    out_dir: str | None,
    logger: CsvLogger,
    display: bool,
    duration: float | None,
    yawn_threshold: float = 0.5,
    yawn_min_duration: float = 0.4,
) -> None:
    """Live emotion testing against a webcam index or RTSP/video URL.

    `source` is passed straight to cv2.VideoCapture, so it can be a
    device index ("0"), a device path, or an RTSP/HTTP URL.
    """
    # Numeric strings ("0", "1", ...) mean a local webcam index.
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open capture source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        # Some webcams/RTSP streams report bogus FPS (0 or absurdly high).
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    name = f"webcam:{source}"

    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "annotated_webcam.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    disp = LazyDisplay(width, height, int(round(fps))) if display else None
    if disp:
        disp.start()

    yawn_detector = YawnDetector(open_threshold=yawn_threshold, min_duration=yawn_min_duration)
    yawn_events = 0
    was_yawning = False

    frame_idx = 0
    start = time.time()
    print(f"[INFO] Streaming from {source} ({width}x{height} @ {fps:.1f}fps). "
          f"{'Close the ffplay window' if display else 'Ctrl+C'} to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed / stream ended.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            scores = _blendshape_dict(result)
            label, confidence = _estimate_emotion(scores)

            t_sec = time.time() - start
            yawning = yawn_detector.update(scores.get("jawOpen", 0.0), t_sec)
            if yawning and not was_yawning:
                yawn_events += 1
                print(f"[{name}] yawn #{yawn_events} started at {t_sec:.1f}s")
            was_yawning = yawning

            logger.log(name, f"{t_sec:.2f}", label, confidence, scores, yawning=yawning)

            annotated = _annotate(frame, label, confidence, scores, yawning=yawning) if (display or writer) else frame

            if writer:
                writer.write(annotated)
            if disp:
                disp.show(annotated)
                if not disp.is_alive():
                    print("[INFO] Display window closed, stopping.")
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[{name}] frame {frame_idx} ({t_sec:.1f}s) -> {label} ({confidence:.2f})")

            if duration is not None and t_sec >= duration:
                print(f"[INFO] Reached --duration {duration}s, stopping.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"    saved -> {out_path}")
        if disp:
            disp.stop()
        elapsed = time.time() - start
        print(f"[{name}] done: {frame_idx} frames in {elapsed:.1f}s, {yawn_events} yawn(s) detected")


def parse_args():
    parser = argparse.ArgumentParser(description="Test the emotion heuristic against images/videos/live camera")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to face_landmarker.task")
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--image-dir", default=None, help="Directory of images to batch-test")
    parser.add_argument("--video", default=None, help="Video file path")
    parser.add_argument(
        "--webcam", default=None,
        help="Live source: webcam device index (e.g. 0) or an RTSP/HTTP stream URL",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="For --webcam: skip the ffplay preview window (still logs CSV / writes video)",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="For --webcam: stop automatically after N seconds",
    )
    parser.add_argument("--out-dir", default=None, help="If set, write annotated images/video here")
    parser.add_argument("--csv", default="emotion_log.csv", help="Path to write blendshape CSV log")
    parser.add_argument(
        "--yawn-threshold", type=float, default=0.5,
        help="jawOpen score above which the mouth counts as 'open' for yawn detection (default: 0.5)",
    )
    parser.add_argument(
        "--yawn-min-duration", type=float, default=0.4,
        help="Seconds jawOpen must stay above threshold to count as a yawn, not just talking (default: 0.4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not any([args.image, args.image_dir, args.video, args.webcam]):
        print("[ERROR] Provide one of --image, --image-dir, --video, or --webcam")
        sys.exit(1)

    _check_model(args.model)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    logger = CsvLogger(args.csv)

    try:
        if args.image or args.image_dir:
            landmarker, mp = _build_face_landmarker(args.model, video_mode=False)

            if args.image:
                process_image(landmarker, mp, args.image, args.out_dir, logger,
                               yawn_threshold=args.yawn_threshold)

            if args.image_dir:
                paths = sorted(
                    p for p in Path(args.image_dir).iterdir()
                    if p.suffix.lower() in IMAGE_EXTS
                )
                if not paths:
                    print(f"[WARN] No images found in {args.image_dir}")
                for p in paths:
                    process_image(landmarker, mp, str(p), args.out_dir, logger,
                                   yawn_threshold=args.yawn_threshold)

        if args.video:
            landmarker, mp = _build_face_landmarker(args.model, video_mode=True)
            process_video(landmarker, mp, args.video, args.out_dir, logger,
                           yawn_threshold=args.yawn_threshold,
                           yawn_min_duration=args.yawn_min_duration)

        if args.webcam:
            landmarker, mp = _build_face_landmarker(args.model, video_mode=True)
            process_webcam(
                landmarker, mp, args.webcam, args.out_dir, logger,
                display=not args.no_display, duration=args.duration,
                yawn_threshold=args.yawn_threshold,
                yawn_min_duration=args.yawn_min_duration,
            )

    finally:
        logger.close()
        print(f"\n[INFO] Blendshape log written to {args.csv}")


if __name__ == "__main__":
    main()