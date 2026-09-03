"""
mar.py
------
Mouth Aspect Ratio (MAR) calculation from MediaPipe face landmarks.
Same idea as EAR, but applied to the mouth to detect yawning.
"""

from typing import Sequence

from scipy.spatial import distance as dist


def mouth_aspect_ratio(landmarks: Sequence, top: int, bottom: int, left: int, right: int, w: int, h: int) -> float:
    """
    Compute the mouth aspect ratio: vertical mouth opening / horizontal mouth width.
    Rises sharply during a yawn and stays low during normal talking or closed-mouth expressions.
    """

    def pt(idx: int):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    vertical = dist.euclidean(pt(top), pt(bottom))
    horizontal = dist.euclidean(pt(left), pt(right))

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def mouth_corner_symmetry(landmarks, top: int, bottom: int, left: int, right: int, w: int, h: int) -> float:
    """
    Where the mouth-corner line sits between the top/bottom lip gap, as a
    ratio of that gap's height: ~0 means centered (a yawn's jaw-drop shape),
    negative means corners lifted toward the top lip (a smile/laugh).

    Experimental filter to reject smiles/laughs that would otherwise pass
    the MAR threshold; validate against real footage before relying on it.
    """

    def y(idx: int):
        return landmarks[idx].y * h

    vertical_gap = y(bottom) - y(top)
    if vertical_gap <= 0:
        return 0.0

    mid_y = (y(top) + y(bottom)) / 2.0
    corner_y = (y(left) + y(right)) / 2.0
    return (corner_y - mid_y) / vertical_gap