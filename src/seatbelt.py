"""
seatbelt.py
-----------
Seatbelt detection via a YOLO NCNN detection model. The model detects the
seatbelt itself, so unlike phone/cigarette (alert on detection), the
violation here is the seatbelt being ABSENT over a sustained window.

A startup grace period keeps the absence buffer from filling up before
the driver has even had a chance to buckle up.
"""

import time
from collections import deque

import cv2
from ultralytics import YOLO

from . import config
from .alerts import log_alert


class SeatbeltDetector:
    def __init__(self):
        self.model = YOLO(config.SEATBELT_MODEL_PATH, task="detect")

        self.seatbelt_class_idx = next(
            (idx for idx, name in self.model.names.items() if name == config.SEATBELT_CLASS_NAME),
            None,
        )
        if self.seatbelt_class_idx is None:
            raise ValueError(
                f"'{config.SEATBELT_CLASS_NAME}' not found in model classes: {self.model.names}"
            )

        self.absent_buffer = deque(maxlen=config.SEATBELT_ABSENT_WINDOW)

        self.last_alert_time = 0.0
        self.alert_count = 0

        self.frame_num = 0
        self.last_results = None
        self.start_time = time.monotonic()

    def process(self, frame, now: float):
        """
        Run (or skip, per SEATBELT_DETECT_EVERY_N_FRAMES) inference on this
        frame and update the absence/cooldown alert state.

        Returns (seatbelt_present, confidence, alert_fired).
        """
        self.frame_num += 1
        should_run_inference = (self.frame_num % config.SEATBELT_DETECT_EVERY_N_FRAMES) == 0

        if should_run_inference:
            self.last_results = self.model.predict(
                source=frame, imgsz=config.SEATBELT_IMG_SIZE, verbose=False, device="cpu"
            )

        confidence = self._max_seatbelt_confidence(self.last_results)
        seatbelt_present = confidence >= config.SEATBELT_CONF_THRESHOLD

        past_grace_period = (time.monotonic() - self.start_time) > config.SEATBELT_STARTUP_GRACE_SEC
        if should_run_inference and past_grace_period:
            self.absent_buffer.append(not seatbelt_present)

        alert_fired = False
        if self._is_absence_confirmed() and (now - self.last_alert_time) > config.SEATBELT_COOLDOWN_SEC:
            self.last_alert_time = now
            self.alert_count += 1
            alert_fired = True
            log_alert(
                event_type="no_seatbelt_detected",
                details={
                    "confidence": round(confidence, 3),
                    "frames_absent": f"{sum(self.absent_buffer)}/{config.SEATBELT_ABSENT_WINDOW}",
                },
            )

        return seatbelt_present, confidence, alert_fired

    def draw(self, frame) -> None:
        """Draw the most recent seatbelt box onto frame, in place."""
        if not self.last_results:
            return

        for box in self.last_results[0].boxes:
            if int(box.cls.item()) != self.seatbelt_class_idx:
                continue
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"seatbelt {confidence:.2f}", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

    def _max_seatbelt_confidence(self, results) -> float:
        if not results:
            return 0.0
        confidences = [
            float(box.conf.item())
            for box in results[0].boxes
            if int(box.cls.item()) == self.seatbelt_class_idx
        ]
        return max(confidences) if confidences else 0.0

    def _is_absence_confirmed(self) -> bool:
        return (
            len(self.absent_buffer) == config.SEATBELT_ABSENT_WINDOW
            and sum(self.absent_buffer) >= config.SEATBELT_ABSENT_FRAMES
        )