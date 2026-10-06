"""
calibration.py
--------------
Per-driver EAR threshold and forward head-pose baseline.

Uses medians rather than means, so blinks (or a quick glance) during the
calibration window don't drag the baseline. The derived EAR threshold is
clamped to a plausible range so a bad calibration (squinting, sunglasses)
can't disable drowsiness detection.

The calibrator is incremental: detector.py fills it during a blocking
startup phase, and if that times out it keeps feeding the same calibrator
from the main loop until it completes. It is also restarted after a long
face absence (possible driver change).
"""

from dataclasses import dataclass
from statistics import median

from . import config


@dataclass
class Baseline:
    ear: float            # driver's typical open-eye EAR
    ear_threshold: float  # closed-eye threshold derived from it
    yaw: float
    pitch: float
    roll: float
    calibrated: bool      # False = config defaults, not measured
    clamped: bool = False


def default_baseline() -> Baseline:
    return Baseline(
        ear=config.EAR_THRESHOLD / config.EAR_THRESHOLD_RATIO,
        ear_threshold=config.EAR_THRESHOLD,
        yaw=0.0, pitch=0.0, roll=0.0,
        calibrated=False,
    )


class BaselineCalibrator:
    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self._samples = []

    def add(self, ear: float, yaw: float, pitch: float, roll: float) -> None:
        if not self.done:
            self._samples.append((ear, yaw, pitch, roll))

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def done(self) -> bool:
        return len(self._samples) >= self.n_samples

    def result(self) -> Baseline:
        ears, yaws, pitches, rolls = zip(*self._samples)
        baseline_ear = float(median(ears))
        raw_threshold = baseline_ear * config.EAR_THRESHOLD_RATIO
        threshold = min(config.EAR_THRESHOLD_MAX, max(config.EAR_THRESHOLD_MIN, raw_threshold))
        return Baseline(
            ear=baseline_ear,
            ear_threshold=threshold,
            yaw=float(median(yaws)),
            pitch=float(median(pitches)),
            roll=float(median(rolls)),
            calibrated=True,
            clamped=threshold != raw_threshold,
        )