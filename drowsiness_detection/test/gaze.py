"""
gaze.py
-------
Head-orientation estimates from MediaPipe face landmarks:
  - head_pose_angles: yaw/pitch/roll from MediaPipe's own facial
    transformation matrix (output_facial_transformation_matrixes=True),
    used for distraction (sustained look-away in any direction). This uses
    MediaPipe's internal pose solve across all face landmarks rather than a
    hand-picked few points, which avoids the correspondence/flip issues a
    manual solvePnP setup is prone to.
  - head_pitch_ratio: 2D vertical ratio, used for sudden head drop (nodding
    off) since that check only needs relative "how far down", not degrees.
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def head_pose_angles(transform_matrix) -> Optional[Tuple[float, float, float]]:
    """
    Decompose MediaPipe's 4x4 facial transformation matrix into yaw/pitch/
    roll degrees. ~0 degrees on each axis means facing forward.
    """
    if transform_matrix is None:
        return None

    rotation_matrix = np.array(transform_matrix)[:3, :3]
    return _euler_from_rotation_matrix(rotation_matrix)


def _euler_from_rotation_matrix(r: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a rotation matrix into yaw/pitch/roll degrees."""
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)

    if sy > 1e-6:
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
        roll = math.atan2(r[2, 1], r[2, 2])
    else:
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0
        roll = math.atan2(-r[1, 2], r[1, 1])

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _euler_from_rotation_matrix(r: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a rotation matrix into yaw/pitch/roll degrees."""
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)

    if sy > 1e-6:
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
        roll = math.atan2(r[2, 1], r[2, 2])
    else:
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0
        roll = math.atan2(-r[1, 2], r[1, 1])

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


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