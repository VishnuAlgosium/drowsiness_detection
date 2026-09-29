"""
vote_window.py
--------------
Time-based voting window for the YOLO detectors: an event is confirmed when
enough of the inferences in the last window_sec seconds agree. Replaces
fixed-length sample buffers, whose real duration changed with FPS.
"""

from collections import deque


class VoteWindow:
    def __init__(self, window_sec: float, min_ratio: float, min_votes: int):
        self.window_sec = window_sec
        self.min_ratio = min_ratio
        self.min_votes = min_votes
        self._votes = deque()  # (timestamp, is_positive) pairs
        self._first_vote_time = None

    def add(self, now: float, is_positive: bool) -> None:
        if self._first_vote_time is None:
            self._first_vote_time = now
        self._votes.append((now, is_positive))
        while now - self._votes[0][0] > self.window_sec:
            self._votes.popleft()

    @property
    def positives(self) -> int:
        return sum(1 for _, is_positive in self._votes if is_positive)

    def __len__(self) -> int:
        return len(self._votes)

    def is_confirmed(self, now: float) -> bool:
        window_filled = (
            self._first_vote_time is not None and now - self._first_vote_time >= self.window_sec
        )
        if not window_filled or len(self._votes) < self.min_votes:
            return False
        return self.positives / len(self._votes) >= self.min_ratio