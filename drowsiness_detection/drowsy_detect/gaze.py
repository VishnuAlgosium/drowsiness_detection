"""
gaze.py
-------
Horizontal head-yaw estimate from MediaPipe face landmarks, used to catch a
driver looking away from the road (side glance) for a sustained period.
"""

from typing import Sequence


def head_yaw_ratio(landmarks: Sequence, nose_idx: int, left_idx: int, right_idx: int) -> float:
    """
    Position of the nose tip between the two face-edge landmarks, as a 0-1
    ratio. ~0.5 means facing forward; it moves toward 0 or 1 as the head
    turns to either side.
    """
    nose_x = landmarks[nose_idx].x
    left_x = landmarks[left_idx].x
    right_x = landmarks[right_idx].x

    face_width = right_x - left_x
    if face_width == 0:
        return 0.5

    return (nose_x - left_x) / face_width