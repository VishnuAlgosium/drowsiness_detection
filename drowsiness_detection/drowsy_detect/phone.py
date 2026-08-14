"""
phone.py
--------
Phone-use detection via a YOLO NCNN model, using confirm-frames + cooldown
alerting. Two confidence tiers:

  - High tier: clear view, fires a "phone_detected" alert.
  - Low tier: occluded/edge-on/calling-position views, needs a longer
    sustained window since a single low-confidence frame is unreliable.
    Fires a "distraction_detected" alert instead of a phone alert, since a
    low-confidence hit is as likely to be some other handheld object.

Inference only runs every config.PHONE_DETECT_EVERY_N_FRAMES frames, since
YOLO on CPU is much heavier than the MediaPipe face landmarker.
"""

import cv2
from collections import deque

from ultralytics import YOLO

from . import config
from .alerts import log_alert


class PhoneDetector:
    def __init__(self):
        self.model = YOLO(config.PHONE_MODEL_PATH, task="detect")

        self.phone_class_idx = next(
            (idx for idx, name in self.model.names.items() if name == config.PHONE_CLASS_NAME),
            None,
        )
        if self.phone_class_idx is None:
            raise ValueError(
                f"'{config.PHONE_CLASS_NAME}' not found in model classes: {self.model.names}"
            )

        self.confirm_buffer = deque(maxlen=config.PHONE_CONFIRM_WINDOW)      # high-confidence tier
        self.low_conf_buffer = deque(maxlen=config.PHONE_LOW_CONF_WINDOW)    # low-confidence tier

        self.last_alert_time = 0.0
        self.alert_count = 0

        self.last_distraction_time = 0.0
        self.distraction_count = 0

        self.frame_num = 0

        # Cached between skipped frames so the on-screen box doesn't flicker.
        self.last_results = None

    def process(self, frame, now: float):
        """
        Run (or skip, per PHONE_DETECT_EVERY_N_FRAMES) inference on this frame
        and update the confirm/cooldown alert state.

        Returns (phone_detected, confidence, phone_alert_fired, distraction_alert_fired).
        """
        self.frame_num += 1
        should_run_inference = (self.frame_num % config.PHONE_DETECT_EVERY_N_FRAMES) == 0

        if should_run_inference:
            self.last_results = self.model.predict(
                source=frame, imgsz=config.PHONE_IMG_SIZE, verbose=False, device="cpu"
            )

        confidence = self._max_phone_confidence(self.last_results)
        phone_detected = confidence >= config.PHONE_CONF_THRESHOLD
        low_conf_detected = confidence >= config.PHONE_LOW_CONF_THRESHOLD

        self.confirm_buffer.append(phone_detected)
        self.low_conf_buffer.append(low_conf_detected)

        phone_alert_fired = False
        if self._is_phone_confirmed() and (now - self.last_alert_time) > config.PHONE_COOLDOWN_SEC:
            self.last_alert_time = now
            self.alert_count += 1
            phone_alert_fired = True
            log_alert(
                event_type="phone_detected",
                details={
                    "confidence": round(confidence, 3),
                    "frames_confirmed": f"{sum(self.confirm_buffer)}/{config.PHONE_CONFIRM_WINDOW}",
                },
            )

        distraction_alert_fired = False
        if self._is_distraction_confirmed() and (now - self.last_distraction_time) > config.PHONE_COOLDOWN_SEC:
            self.last_distraction_time = now
            self.distraction_count += 1
            distraction_alert_fired = True
            log_alert(
                event_type="distraction_detected",
                details={
                    "source": "phone_low_confidence",
                    "confidence": round(confidence, 3),
                    "frames_confirmed_low_conf": f"{sum(self.low_conf_buffer)}/{config.PHONE_LOW_CONF_WINDOW}",
                },
            )

        return phone_detected, confidence, phone_alert_fired, distraction_alert_fired

    def draw(self, frame) -> None:
        """Draw the most recent phone box(es) onto frame, in place."""
        if not self.last_results:
            return

        for box in self.last_results[0].boxes:
            if int(box.cls.item()) != self.phone_class_idx:
                continue
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, f"phone {confidence:.2f}", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )

    def _max_phone_confidence(self, results) -> float:
        if not results:
            return 0.0
        confidences = [
            float(box.conf.item())
            for box in results[0].boxes
            if int(box.cls.item()) == self.phone_class_idx
        ]
        return max(confidences) if confidences else 0.0

    def _is_phone_confirmed(self) -> bool:
        return (
            len(self.confirm_buffer) == config.PHONE_CONFIRM_WINDOW
            and sum(self.confirm_buffer) >= config.PHONE_CONFIRM_FRAMES
        )

    def _is_distraction_confirmed(self) -> bool:
        return (
            len(self.low_conf_buffer) == config.PHONE_LOW_CONF_WINDOW
            and sum(self.low_conf_buffer) >= config.PHONE_LOW_CONF_FRAMES
        )