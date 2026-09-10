"""
rtsp_lag_check.py
------------------
Quick standalone check for an RTSP camera's:
  - actual delivered FPS (frames/sec your process is receiving)
  - end-to-end lag (how far behind "now" the frames are)

Usage:
    python rtsp_lag_check.py rtsp://user:pass@192.168.1.50:554/stream1

Notes on the lag measurement:
  There's no way to get the camera's own encode timestamp from a plain
  RTSP/H264 stream without parsing RTCP sender reports, which OpenCV
  doesn't expose. So this script uses a practical proxy instead: it
  overlays the wall-clock time it received each frame, and pipes video
  to ffplay (via LazyDisplay, same as production) so you can hold a
  second clock/phone in view of the camera and visually compare the
  overlaid time vs the real clock. That difference is your true
  end-to-end lag.

  What THIS script gives you automatically, without any manual step:
    - Frames per second actually being delivered to cv2.read()
    - How long each cap.read() call takes (spikes = network/decode stalls)
    - Whether frames are being silently dropped (grab() failing)

  This machine has no GUI backend (headless OpenCV build -- cv2.imshow
  is not implemented here), so display goes through ffplay via
  LazyDisplay, exactly like the rest of this project, instead of
  cv2.imshow/cv2.waitKey.

Press Ctrl+C in the terminal to quit.
"""

import os
import sys
import time
import cv2

# Allow running this file directly by putting the project root -- two
# levels up from this file, e.g. drowsiness_detection/test/ -> repo
# root -- on sys.path, so `drowsiness_detection.drowsy_detect...` is
# importable regardless of current working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from drowsiness_detection.drowsy_detect.display import LazyDisplay

def main():
    if len(sys.argv) < 2:
        print("Usage: python rtsp_lag_check.py <rtsp_url>")
        sys.exit(1)

    rtsp_url = sys.argv[1]

    print(f"[INFO] Connecting to: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # Keep the internal buffer as small as possible so we're always
    # looking at the freshest frame, not a queued-up backlog.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Could not open RTSP stream. Check URL/credentials/network.")
        sys.exit(1)

    # Detected frame size, needed so LazyDisplay's ffplay subprocess is
    # started with the exact -video_size the incoming frames will be.
    # LazyDisplay/ffplay is picky: every frame piped to it must match
    # the size it was started with exactly, or ffplay silently fails.
    ret, probe_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read an initial frame to detect size.")
        sys.exit(1)
    h, w = probe_frame.shape[:2]
    fps_hint = cap.get(cv2.CAP_PROP_FPS) or 15
    print(f"[INFO] Detected frame size: {w}x{h}, reported FPS: {fps_hint:.1f}")

    display = LazyDisplay(w, h, int(fps_hint))
    display.start()

    print("[INFO] Connected. Press Ctrl+C in the terminal to quit.\n")

    frame_count = 0
    fps_window_start = time.time()
    fps_window_count = 0

    read_times = []  # rolling list of how long each cap.read() took

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            read_duration = time.perf_counter() - t0

            if not ret:
                print("[WARN] Frame grab failed -- stream may have dropped.")
                time.sleep(0.5)
                continue

            now = time.time()
            frame_count += 1
            fps_window_count += 1
            read_times.append(read_duration)

            # Print a rolling FPS + read-time report once a second.
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                fps = fps_window_count / elapsed
                avg_read_ms = (sum(read_times) / len(read_times)) * 1000
                max_read_ms = max(read_times) * 1000
                print(f"[STATS] FPS: {fps:5.1f}  |  "
                      f"avg read: {avg_read_ms:6.1f} ms  |  "
                      f"max read: {max_read_ms:6.1f} ms  |  "
                      f"total frames: {frame_count}")
                fps_window_start = now
                fps_window_count = 0
                read_times.clear()

            # Overlay current wall-clock time on the frame. Hold a
            # second clock/phone in view of the camera and compare
            # what's overlaid here vs what the physical clock shows --
            # the difference is your true end-to-end lag.
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(now)) + \
                f".{int((now % 1) * 1000):03d}"
            cv2.putText(frame, f"Recv: {timestamp_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            display.show(frame)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        display.stop()
        print(f"[INFO] Done. Total frames received: {frame_count}")

if __name__ == "__main__":
    main()