"""
phone.py
--------
Phone-use detection via a YOLO ONNX model, using the same
confirm-frames + cooldown alerting pattern as the drowsiness/yawn detectors.

Inference only runs every config.PHONE_DETECT_EVERY_N_FRAMES frames, since
YOLO on CPU is much heavier than the MediaPipe face landmarker and running
it every frame is what makes the shared capture/display loop lag.
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

        self.confirm_buffer = deque(maxlen=config.PHONE_CONFIRM_WINDOW)
        self.last_alert_time = 0.0
        self.alert_count = 0
        self.frame_num = 0

        # Cached between skipped frames so the on-screen box doesn't flicker
        # away on frames where inference didn't actually run.
        self.last_results = None

    def process(self, frame, now: float):
        """
        Run (or skip, per PHONE_DETECT_EVERY_N_FRAMES) inference on this frame
        and update the confirm/cooldown alert state.

        Returns (phone_detected, confidence, alert_just_fired).
        """
        self.frame_num += 1
        should_run_inference = (self.frame_num % config.PHONE_DETECT_EVERY_N_FRAMES) == 0

        if should_run_inference:
            self.last_results = self.model.predict(
                source=frame, imgsz=config.PHONE_IMG_SIZE, verbose=False, device="cpu"
            )

        confidence = self._max_phone_confidence(self.last_results)
        phone_detected = confidence >= config.PHONE_CONF_THRESHOLD
        self.confirm_buffer.append(phone_detected)

        alert_just_fired = False
        if self._is_confirmed() and (now - self.last_alert_time) > config.PHONE_COOLDOWN_SEC:
            self.last_alert_time = now
            self.alert_count += 1
            alert_just_fired = True
            log_alert(
                event_type="phone_detected",
                details={
                    "confidence": round(confidence, 3),
                    "frames_confirmed": f"{sum(self.confirm_buffer)}/{config.PHONE_CONFIRM_WINDOW}",
                },
            )

        return phone_detected, confidence, alert_just_fired

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

    def _is_confirmed(self) -> bool:
        return (
            len(self.confirm_buffer) == config.PHONE_CONFIRM_WINDOW
            and sum(self.confirm_buffer) >= config.PHONE_CONFIRM_FRAMES
        )
