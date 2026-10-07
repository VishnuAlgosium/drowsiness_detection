"""
phone.py
--------
Phone-use detection via a YOLO NCNN model, using confirm-frames + cooldown
alerting. Two confidence tiers:

  - High tier: clear view, fires a "phone_detected" alert.
  - Low tier: occluded/edge-on/calling-position views, needs a longer
    time window and fires a "distraction_detected" alert instead,
    since a single low-confidence hit is as likely to be another object.

Inference only runs every config.PHONE_DETECT_EVERY_N_FRAMES frames, since
YOLO on CPU is much heavier than the MediaPipe face landmarker.
"""

import math

import cv2
from ultralytics import YOLO

from . import config
from .alerts import log_alert
from .vote_window import VoteWindow


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

        self.confirm_votes = VoteWindow(
            config.PHONE_CONFIRM_WINDOW_SEC, config.PHONE_CONFIRM_RATIO, config.DETECTOR_MIN_VOTES
        )
        self.low_conf_votes = VoteWindow(
            config.PHONE_LOW_CONF_WINDOW_SEC, config.PHONE_LOW_CONF_RATIO, config.DETECTOR_MIN_VOTES
        )

        self.last_alert_time = -math.inf
        self.alert_count = 0

        self.last_distraction_time = -math.inf
        self.distraction_count = 0

        self.frame_num = 0
        self.last_results = None  # cached between skipped frames so the box doesn't flicker
        self.last_confidence = 0.0  # cached alongside last_results, avoids recomputing on skipped frames

    def process(self, frame, now: float):
        """
        Run (or skip, per PHONE_DETECT_EVERY_N_FRAMES) inference on this frame
        and update the confirm/cooldown alert state.

        Returns (phone_detected, confidence, phone_alert_fired, distraction_alert_fired).
        """
        self.frame_num += 1
        should_run_inference = (
            self.frame_num % config.PHONE_DETECT_EVERY_N_FRAMES
        ) == config.PHONE_DETECT_OFFSET

        if should_run_inference:
            self.last_results = self.model.predict(
                source=frame, imgsz=config.PHONE_IMG_SIZE, verbose=False, device="cpu"
            )
            self.last_confidence = self._max_phone_confidence(self.last_results)

        confidence = self.last_confidence
        phone_detected = confidence >= config.PHONE_CONF_THRESHOLD
        low_conf_detected = confidence >= config.PHONE_LOW_CONF_THRESHOLD

        # Only record a sample on frames where inference actually ran, so a
        # single detection isn't counted once per skipped frame too.
        if should_run_inference:
            self.confirm_votes.add(now, phone_detected)
            self.low_conf_votes.add(now, low_conf_detected)

        phone_alert_fired = False
        if self.confirm_votes.is_confirmed(now) and (now - self.last_alert_time) > config.PHONE_COOLDOWN_SEC:
            self.last_alert_time = now
            self.alert_count += 1
            phone_alert_fired = True
            log_alert(
                event_type="phone_detected",
                details={
                    "confidence": round(confidence, 3),
                    "frames_confirmed": f"{self.confirm_votes.positives}/{len(self.confirm_votes)}",
                },
            )

        distraction_alert_fired = False
        if self.low_conf_votes.is_confirmed(now) and (now - self.last_distraction_time) > config.PHONE_COOLDOWN_SEC:
            self.last_distraction_time = now
            self.distraction_count += 1
            distraction_alert_fired = True
            log_alert(
                event_type="distraction_detected",
                details={
                    "source": "phone_low_confidence",
                    "confidence": round(confidence, 3),
                    "frames_confirmed_low_conf": f"{self.low_conf_votes.positives}/{len(self.low_conf_votes)}",
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