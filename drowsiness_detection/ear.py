"""Eye Aspect Ratio (EAR) computation from MediaPipe Face Mesh landmarks."""

import numpy as np
from scipy.spatial import distance as dist


def eye_aspect_ratio(landmarks, eye_indices, img_w: int, img_h: int) -> float:
    """
    Compute Eye Aspect Ratio from 6 landmark indices.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = np.array(
        [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices],
        dtype=np.float32
    )
    v1 = dist.euclidean(pts[1], pts[5])
    v2 = dist.euclidean(pts[2], pts[4])
    h  = dist.euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
