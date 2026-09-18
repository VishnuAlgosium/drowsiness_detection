"""
capture.py
----------
Background camera-capture thread.

cv2.VideoCapture.read() can block for a variable, bursty amount of time
depending on the camera driver/USB stack -- long waits followed by several
already-buffered frames returning almost instantly. Calling it inline in
the main loop ties processing latency (and the per-frame FPS numbers) to
that jitter. FrameGrabber reads continuously on its own thread and hands
the main loop whichever frame is most recent, so a capture stall no longer
stalls detection/display, and a capture burst doesn't get processed twice.
"""

import threading
import time


class FrameGrabber:
    """Continuously reads frames from a cv2.VideoCapture in the background."""

    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._last_frame_time = time.monotonic()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._last_frame_time = time.monotonic()
            else:
                time.sleep(0.01)  # avoid a hot loop while the camera is down

    def read(self):
        """Returns (frame, frame_id, seconds_since_last_frame). frame_id lets
        the caller skip re-processing the same frame while waiting for the
        next one; seconds_since_last_frame lets it detect a dead camera."""
        with self._lock:
            age = time.monotonic() - self._last_frame_time
            return self._frame, self._frame_id, age

    def release(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)