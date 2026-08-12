"""
capture_frames.py
------------------
Grabs a frame every N seconds from a video source and saves it to a folder.

Usage:
    python capture_frames.py
"""

import os
import time

import cv2

SOURCE = 2                      # 0/1/2 for webcam, or a video/RTSP path
INTERVAL_SEC = 0.5              # seconds between saved frames
OUTPUT_DIR = "captured_frames"

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open source: {SOURCE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

frame_num = 0
saved_count = 0
last_save_time = 0.0

print(f"[INFO] Saving a frame every {INTERVAL_SEC}s to '{OUTPUT_DIR}/' (press 'q' or Ctrl+C to stop)")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[INFO] No more frames or camera not returning frames.")
            break

        frame_num += 1
        now = time.time()

        if now - last_save_time >= INTERVAL_SEC:
            filename = os.path.join(OUTPUT_DIR, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            last_save_time = now
            print(f"[SAVED] {filename}")

        cv2.imshow("Frame Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Stopped by user")
            break

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. {saved_count} frames saved to '{OUTPUT_DIR}/'")