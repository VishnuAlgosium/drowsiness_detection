"""
perclos.py
----------
PERCLOS (PERcentage of eye CLOSure): the fraction of the last window_sec
spent with eyes closed. Unlike the consecutive-frame counter in detector.py,
this rises smoothly as blinks get longer or more frequent, so it can catch
a driver fading out well before any single closure is long enough to trip
that counter.
"""

from collections import deque


class PerclosMonitor:
    """Tracks recent closed/open frames to compute a rolling PERCLOS score."""

    def __init__(self, window_sec: float):
        self.window_sec = window_sec
        self._samples = deque()  # (timestamp, eyes_closed) pairs

    def update(self, now: float, eyes_closed: bool) -> float:
        """Feed one frame's eye state and return the current PERCLOS (0-1)."""
        self._samples.append((now, eyes_closed))

        while self._samples and now - self._samples[0][0] > self.window_sec:
            self._samples.popleft()

        closed_count = sum(1 for _, closed in self._samples if closed)
        return closed_count / len(self._samples)

    def reset(self) -> None:
        """Clear state, e.g. when the face is lost."""
        self._samples.clear()