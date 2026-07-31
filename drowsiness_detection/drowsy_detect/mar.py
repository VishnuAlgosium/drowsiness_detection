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

    Rises sharply during a yawn (wide, sustained mouth opening) and stays low
    during normal talking or closed-mouth expressions.
    """

    def pt(idx: int):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    vertical = dist.euclidean(pt(top), pt(bottom))
    horizontal = dist.euclidean(pt(left), pt(right))

    if horizontal == 0:
        return 0.0

    return vertical / horizontal
