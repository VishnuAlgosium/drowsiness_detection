#!/usr/bin/env python
"""
Simple standalone drowsiness detector.

- Detects drowsiness via EAR (Eye Aspect Ratio) using MediaPipe Face Mesh
- Plays an alert beep when eyes stay closed too long
- Display can be turned ON/OFF at runtime (press 'v') to save CPU —
  when OFF, no frame is drawn or piped to ffplay at all
- No images/screenshots are ever saved to disk

Run:   python simple_drowsiness.py
Keys (typed in THIS terminal window):
  v = toggle video display on/off
  q = quit
"""

import os
import sys
import time
import select
import termios
import tty
import subprocess
import shutil

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

# ── Force CPU-only ───────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(4)

# ── Config ───────────────────────────────────────────────────────────────────
EAR_THRESHOLD      = 0.25
CONSEC_FRAMES      = 48
ALERT_COOLDOWN_SEC = 4.0
DISPLAY_ON_START   = True   # starting state; toggle live with 'v'

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30
DISPLAY_W, DISPLAY_H = 640, 480


# ── Audio (mono beep, fixed for pygame mono mixer) ──────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    AUDIO_AVAILABLE = True
except Exception as e:
    AUDIO_AVAILABLE = False
    print(f"[WARN] pygame audio unavailable: {e}")


def _beep(freq=880.0, dur=0.28, vol=0.6, sr=44100):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    env = np.exp(-t * 8)
    return (wave * env * vol * 32767).astype(np.int16)


def play_alert():
    if not AUDIO_AVAILABLE:
        return
    try:
        sr = 44100
        silence = np.zeros(int(sr * 0.1), dtype=np.int16)
        b1, b2 = _beep(880), _beep(660)
        seq = np.concatenate([b1, silence, b1, silence, b2])
        pygame.sndarray.make_sound(seq).play()
    except Exception as e:
        print(f"[WARN] Audio error: {e}")


# ── EAR ──────────────────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx], dtype=np.float32)
    v1 = dist.euclidean(pts[1], pts[5])
    v2 = dist.euclidean(pts[2], pts[4])
    hh = dist.euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * hh) if hh > 0 else 0.0


# ── Non-blocking terminal key reader ────────────────────────────────────────
class KeyReader:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._old = None
        if self.enabled:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def get_key(self):
        if not self.enabled:
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if r else None

    def restore(self):
        if self.enabled and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


# ── Lazy ffplay display (only started when display is turned ON) ───────────
class LazyDisplay:
    def __init__(self, width, height, fps=30):
        self.width, self.height, self.fps = width, height, fps
        self.proc = None

    def start(self):
        if self.proc is not None:
            return
        if shutil.which("ffplay") is None:
            print("[WARN] ffplay not found — install with: sudo apt-get install -y ffmpeg")
            return
        cmd = [
            "ffplay", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-fflags", "nobuffer", "-flags", "low_delay", "-framedrop",
            "-an", "-sn", "-window_title", "Drowsiness Detection", "-i", "-",
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def show(self, frame):
        if self.proc is None:
            return True
        if self.proc.poll() is not None:
            self.proc = None
            return True
        try:
            self.proc.stdin.write(frame.tobytes())
            return True
        except (BrokenPipeError, OSError):
            self.proc = None
            return True

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=1)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    display = LazyDisplay(DISPLAY_W, DISPLAY_H, fps=30)
    display_on = DISPLAY_ON_START
    if display_on:
        display.start()

    keys = KeyReader()
    counter = 0
    alert_count = 0
    last_alert = 0.0
    no_face_streak = 0
    NO_FACE_GRACE = 6   # tolerate up to N consecutive no-face frames (IR flicker) before penalizing
    max_streak_seen = 0
    last_streak_report = 0.0

    print("\n[INFO] Running. Keys: v = toggle display, q = quit\n")
    print(f"[INFO] Display currently: {'ON' if display_on else 'OFF'}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame grab failed.")
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Only convert/copy for display if display is on — saves CPU when off
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            current_ear = 0.0
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                ear_l = eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h)
                ear_r = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
                current_ear = (ear_l + ear_r) / 2.0
                counter = counter + 1 if current_ear < EAR_THRESHOLD else max(0, counter - 1)
                if no_face_streak > 0:
                    max_streak_seen = max(max_streak_seen, no_face_streak)
                no_face_streak = 0
            else:
                no_face_streak += 1
                if no_face_streak > NO_FACE_GRACE:
                    counter = max(0, counter - 2)  # only penalize once flicker becomes a real gap
                # else: hold counter steady — likely just an IR flicker/exposure blip

            now = time.time()
            if counter >= CONSEC_FRAMES:
                if now - last_alert > ALERT_COOLDOWN_SEC:
                    alert_count += 1
                    print(f"[ALERT #{alert_count}] Drowsiness detected at {time.strftime('%H:%M:%S')} "
                          f"(EAR={current_ear:.3f})")
                    play_alert()
                    last_alert = now

            # Report worst flicker streak seen, once per second (only if any misses happened)
            if now - last_streak_report > 1.0:
                if max_streak_seen > 0:
                    print(f"[FLICKER] Longest no-face streak in last 1s: {max_streak_seen} frame(s) "
                          f"(grace tolerance = {NO_FACE_GRACE})")
                max_streak_seen = 0
                last_streak_report = now

            # Only draw/send frame to display if display is currently ON
            if display_on:
                label = f"EAR: {current_ear:.3f}  Counter: {counter}/{CONSEC_FRAMES}  Alerts: {alert_count}"
                color = (0, 0, 255) if counter >= CONSEC_FRAMES else (0, 255, 0)
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H:
                    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                display.show(frame)

            key = keys.get_key()
            if key:
                key = key.lower()
                if key == 'q':
                    print("\n[INFO] Quitting...")
                    break
                elif key == 'v':
                    display_on = not display_on
                    if display_on:
                        display.start()
                        print("[INFO] Display turned ON")
                    else:
                        display.stop()
                        print("[INFO] Display turned OFF (saving CPU)")

    finally:
        keys.restore()
        display.stop()
        face_mesh.close()
        cap.release()
        if AUDIO_AVAILABLE:
            pygame.mixer.quit()

    print(f"\n[INFO] Session ended. Total alerts: {alert_count}")


if __name__ == "__main__":
    main()