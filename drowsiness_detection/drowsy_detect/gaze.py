"""
gaze.py
-------
Head-orientation estimates from MediaPipe face landmarks:
  - head_yaw_ratio:   horizontal turn, used to catch a sustained side glance.
  - head_pitch_ratio: vertical tilt, used to catch a sudden head drop (nodding off).
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


def head_pitch_ratio(landmarks: Sequence, forehead_idx: int, chin_idx: int, nose_idx: int) -> float:
    """
    Position of the nose tip between the forehead and chin, as a 0-1 ratio.
    ~0.5 means facing forward; it rises toward 1 as the head tips down.
    """
    forehead_y = landmarks[forehead_idx].y
    chin_y = landmarks[chin_idx].y
    nose_y = landmarks[nose_idx].y

    face_height = chin_y - forehead_y
    if face_height == 0:
        return 0.5

    return (nose_y - forehead_y) / face_height