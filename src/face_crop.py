"""
face_crop.py
------------
Padded face bounding box from MediaPipe landmarks, used to crop the frame
before cigarette classification (isolates mouth/face area from background).
"""

from typing import Optional, Sequence, Tuple


def padded_face_box(landmarks: Sequence, w: int, h: int, padding_ratio: float) -> Tuple[int, int, int, int]:
    """Bounding box around all landmarks, expanded by padding_ratio on each side."""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    return x1, y1, x2, y2