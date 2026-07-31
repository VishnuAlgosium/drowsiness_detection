"""
alerts.py
---------
Shared logging for all three detectors (phone, drowsiness, yawn):

- log_alert(): one JSONL line per alert, for an audit trail of events.
- FrameLogger: one CSV row per frame with all three signals, so QA can
  tune thresholds after the fact by looking at raw EAR/MAR/phone-confidence
  values instead of just the alert moments.
"""

import csv
import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np

from . import config


def _json_safe(value):
    """json.dumps doesn't know how to serialize numpy scalar types (e.g. the
    float32 that ear.py/mar.py return) -- convert those to plain Python floats."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def log_alert(event_type: str, details: dict) -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(
        config.LOG_DIR, f"{config.CAMERA_ID}_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    )

    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event_type": event_type,
        **details,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(event, default=_json_safe) + "\n")


class FrameLogger:
    """Per-frame CSV of all three signals, written once per loop iteration."""

    def __init__(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.path = os.path.join(
            config.LOG_DIR, f"frames_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["timestamp", "frame_num", "ear", "mar", "phone_confidence", "fps"]
        )

    def write(self, frame_num: int, ear: float, mar: float, phone_confidence: float, fps: float) -> None:
        self._writer.writerow([
            datetime.now().isoformat(),
            frame_num,
            f"{ear:.3f}",
            f"{mar:.3f}",
            f"{phone_confidence:.3f}",
            f"{fps:.1f}",
        ])

    def close(self) -> None:
        self._file.close()