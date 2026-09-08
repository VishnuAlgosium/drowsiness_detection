"""
blink_visibility.py
--------------------
Flags when the eyes look hidden from the camera (e.g. sunglasses), so
EAR-based drowsiness detection can be trusted or not.

Core idea: a blink -- EAR dipping below the closed threshold and quickly
recovering -- is only possible if the camera can actually see the eyelid
move. If no blink shows up over a long window, the eyes are either hidden
from the camera or in sustained closure (already caught separately by the
drowsiness counter), so treat "no blinks for a while" as "can't trust EAR."
"""

from collections import deque


class BlinkVisibilityMonitor:
    """Tracks recent blink timestamps to flag when eye state can't be trusted."""

    def __init__(self, no_blink_timeout_sec: float, blink_ear_threshold: float, max_blink_frames: int):
        self.no_blink_timeout_sec = no_blink_timeout_sec
        self.blink_ear_threshold = blink_ear_threshold
        self.max_blink_frames = max_blink_frames  # longer closures are sustained closure, not a blink

        self._blink_times = deque()
        self._closed_run = 0
        self._monitor_start = None

    def update(self, now: float, ear: float) -> bool:
        """Feed one frame's EAR and return whether the eyes look hidden."""
        if self._monitor_start is None:
            self._monitor_start = now

        eyes_closed = ear < self.blink_ear_threshold
        if eyes_closed:
            self._closed_run += 1
        else:
            if 0 < self._closed_run <= self.max_blink_frames:
                self._blink_times.append(now)  # short closure that recovered = a blink
            self._closed_run = 0

        while self._blink_times and now - self._blink_times[0] > self.no_blink_timeout_sec:
            self._blink_times.popleft()

        warmed_up = now - self._monitor_start >= self.no_blink_timeout_sec
        return warmed_up and len(self._blink_times) == 0

    def reset(self) -> None:
        """Clear state, e.g. when the face is lost."""
        self._blink_times.clear()
        self._closed_run = 0
        self._monitor_start = None