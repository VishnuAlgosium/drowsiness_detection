"""
perclos.py
----------
PERCLOS (PERcentage of eye CLOSure): the fraction of the last window_sec
spent with eyes closed. Unlike the closure timer in detector.py, this rises
smoothly as blinks get longer or more frequent, so it can catch a driver
fading out well before any single closure is long enough to trip that timer.

Time-weighted (each sample counts for its frame's dt), so a slow stretch of
frames doesn't count for less than a fast one. `ready` stays False until
min_coverage_sec of eye-state time is in the window -- otherwise one closed
frame right after startup or a reset reads as 100%.
"""

from collections import deque


class PerclosMonitor:
    """Tracks recent closed/open time to compute a rolling PERCLOS score."""

    def __init__(self, window_sec: float, min_coverage_sec: float):
        self.window_sec = window_sec
        self.min_coverage_sec = min_coverage_sec
        self._samples = deque()  # (timestamp, eyes_closed, dt) triples
        self._total_sec = 0.0
        self._closed_sec = 0.0

    def update(self, now: float, eyes_closed: bool, dt: float) -> float:
        """Feed one frame's eye state (lasting dt seconds); return current PERCLOS (0-1)."""
        self._samples.append((now, eyes_closed, dt))
        self._total_sec += dt
        if eyes_closed:
            self._closed_sec += dt
        self.evict(now)
        return self.value

    def evict(self, now: float) -> None:
        """Drop samples older than the window. Safe to call on frames with no update
        (e.g. face lost), so paused data still ages out on real time."""
        while self._samples and now - self._samples[0][0] > self.window_sec:
            _, closed, dt = self._samples.popleft()
            self._total_sec -= dt
            if closed:
                self._closed_sec -= dt
        if not self._samples:  # clear accumulated float drift
            self._total_sec = self._closed_sec = 0.0

    @property
    def value(self) -> float:
        if self._total_sec <= 0:
            return 0.0
        return max(0.0, min(1.0, self._closed_sec / self._total_sec))

    @property
    def coverage_sec(self) -> float:
        return max(0.0, self._total_sec)

    @property
    def ready(self) -> bool:
        return self.coverage_sec >= self.min_coverage_sec

    def reset(self) -> None:
        """Clear state, e.g. after a long face absence."""
        self._samples.clear()
        self._total_sec = self._closed_sec = 0.0