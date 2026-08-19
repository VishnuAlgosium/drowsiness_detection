"""
yawn_webcam.py
--------------
Minimal webcam-only yawn detector. Uses ONLY the jawOpen blendshape
from MediaPipe FaceLandmarker — sustained-open-duration heuristic.

Usage:
    python yawn_webcam.py --webcam 0
    python yawn_webcam.py --webcam rtsp://user:pass@host/stream
"""

import argparse
import shutil
import subprocess
import sys
import time
import os

import cv2
import numpy as np

DEFAULT_MODEL_PATH = "drowsiness_detection/models/face_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _check_model(model_path: str) -> None:
    if not os.path.exists(model_path):
        print(f"[ERROR] Missing {model_path}")
        print(f"Download it first:\nwget -O {model_path} {MODEL_DOWNLOAD_URL}")
        sys.exit(1)


def _build_face_landmarker(model_path: str):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options), mp


def _get_jaw_open(face_results) -> float:
    if not face_results.face_blendshapes:
        return 0.0
    for b in face_results.face_blendshapes[0]:
        if b.category_name == "jawOpen":
            return b.score
    return 0.0


class YawnDetector:
    """Flags a yawn from a SUSTAINED high jawOpen score, not a single frame."""

    def __init__(self, open_threshold: float = 0.5, min_duration: float = 0.4):
        self.open_threshold = open_threshold
        self.min_duration = min_duration
        self._open_since = None
        self.is_yawning = False

    def update(self, jaw_open_score: float, timestamp_sec: float) -> bool:
        if jaw_open_score >= self.open_threshold:
            if self._open_since is None:
                self._open_since = timestamp_sec
            self.is_yawning = (timestamp_sec - self._open_since) >= self.min_duration
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


def _annotate(frame: np.ndarray, jaw_open: float, yawning: bool) -> np.ndarray:
    out = frame.copy()
    color = (0, 0, 255) if yawning else (0, 255, 0)
    cv2.putText(out, f"jawOpen: {jaw_open:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if yawning:
        cv2.putText(out, "YAWNING", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out


def process_webcam(landmarker, mp, source: str, yawn_threshold: float, yawn_min_duration: float,
                    display: bool = True):
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open capture source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    disp = LazyDisplay(width, height, int(round(fps))) if display else None
    if disp:
        disp.start()

    yawn_detector = YawnDetector(open_threshold=yawn_threshold, min_duration=yawn_min_duration)
    yawn_events = 0
    was_yawning = False
    start = time.time()

    print(f"[INFO] Streaming from {source}. "
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

            jaw_open = _get_jaw_open(result)
            t_sec = time.time() - start
            yawning = yawn_detector.update(jaw_open, t_sec)

            if yawning and not was_yawning:
                yawn_events += 1
                print(f"[YAWN #{yawn_events}] started at {t_sec:.1f}s (jawOpen={jaw_open:.2f})")
            was_yawning = yawning

            if disp:
                annotated = _annotate(frame, jaw_open, yawning)
                disp.show(annotated)
                if not disp.is_alive():
                    print("[INFO] Display window closed, stopping.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if disp:
            disp.stop()
        elapsed = time.time() - start
        print(f"[DONE] {elapsed:.1f}s elapsed, {yawn_events} yawn(s) detected")


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal webcam yawn detector")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to face_landmarker.task")
    parser.add_argument("--webcam", required=True, help="Webcam device index or RTSP/HTTP URL")
    parser.add_argument("--yawn-threshold", type=float, default=0.5,
                         help="jawOpen score above which mouth counts as 'open' (default: 0.5)")
    parser.add_argument("--yawn-min-duration", type=float, default=0.4,
                         help="Seconds jawOpen must stay open to count as a yawn (default: 0.4)")
    parser.add_argument("--no-display", action="store_true",
                         help="Skip the ffplay preview window (still prints yawn events)")
    return parser.parse_args()


def main():
    args = parse_args()
    _check_model(args.model)
    landmarker, mp = _build_face_landmarker(args.model)
    process_webcam(landmarker, mp, args.webcam, args.yawn_threshold, args.yawn_min_duration,
                    display=not args.no_display)


if __name__ == "__main__":
    main()