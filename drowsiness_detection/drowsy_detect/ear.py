"""
ear.py
------
Eye Aspect Ratio (EAR) calculation from MediaPipe face landmarks.
"""

from typing import Sequence

import numpy as np
from scipy.spatial import distance as dist


def eye_aspect_ratio(landmarks: Sequence, idx: Sequence[int], w: int, h: int) -> float:
    """
    Compute the eye aspect ratio for one eye.

    idx must be six landmark indices ordered as:
    [outer_corner, top_1, top_2, inner_corner, bottom_1, bottom_2]
    """
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in idx],
        dtype=np.float32,
    )

    v1 = dist.euclidean(pts[1], pts[5])
    v2 = dist.euclidean(pts[2], pts[4])
    hh = dist.euclidean(pts[0], pts[3])

    if hh == 0:
        return 0.0

    return (v1 + v2) / (2.0 * hh)


def average_ear(landmarks: Sequence, left_idx: Sequence[int], right_idx: Sequence[int], w: int, h: int) -> float:
    """Average EAR across both eyes."""
    left_ear = eye_aspect_ratio(landmarks, left_idx, w, h)
    right_ear = eye_aspect_ratio(landmarks, right_idx, w, h)
    return (left_ear + right_ear) / 2.0
