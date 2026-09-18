"""
cigarette.py
------------
Cigarette-use detection via a YOLO NCNN classification model, using the
same confirm-frames + cooldown alerting as phone.py. Unlike phone/seatbelt,
this model classifies a cropped face region ("cigarette" vs "nocigarette")
rather than detecting a box, so there's nothing to draw and no tiered
confidence -- just one confirm buffer.

Inference only runs every config.CIGARETTE_DETECT_EVERY_N_FRAMES frames,
same reasoning as phone.py.
"""

from collections import deque

from ultralytics import YOLO

from . import config
from .alerts import log_alert


class CigaretteDetector:
    def __init__(self):
        self.model = YOLO(config.CIGARETTE_MODEL_PATH, task="classify")

        self.cigarette_class_idx = next(
            (idx for idx, name in self.model.names.items() if name == config.CIGARETTE_CLASS_NAME),
            None,
        )
        if self.cigarette_class_idx is None:
            raise ValueError(
                f"'{config.CIGARETTE_CLASS_NAME}' not found in model classes: {self.model.names}"
            )

        self.confirm_buffer = deque(maxlen=config.CIGARETTE_CONFIRM_WINDOW)

        self.last_alert_time = 0.0
        self.alert_count = 0

        self.frame_num = 0
        self.last_confidence = 0.0

    def process(self, face_crop, now: float):
        """
        Run (or skip, per CIGARETTE_DETECT_EVERY_N_FRAMES) classification on
        the cropped face region and update the confirm/cooldown alert state.

        face_crop is None (or empty) when no face was found this frame --
        treated as "no cigarette" for that frame, since there's nothing to
        classify.

        Returns (cigarette_detected, confidence, alert_fired).
        """
        self.frame_num += 1
        should_run_inference = (
            self.frame_num % config.CIGARETTE_DETECT_EVERY_N_FRAMES
        ) == config.CIGARETTE_DETECT_OFFSET

        if should_run_inference:
            if face_crop is not None and face_crop.size > 0:
                results = self.model.predict(
                    source=face_crop, imgsz=config.CIGARETTE_IMG_SIZE, verbose=False, device="cpu"
                )
                # Read the cigarette class's own probability directly, rather than
                # only when it happens to be the top-1 class.
                self.last_confidence = float(results[0].probs.data[self.cigarette_class_idx])
            else:
                self.last_confidence = 0.0

        cigarette_detected = self.last_confidence >= config.CIGARETTE_CONF_THRESHOLD

        if should_run_inference:
            self.confirm_buffer.append(cigarette_detected)

        alert_fired = False
        if self._is_confirmed() and (now - self.last_alert_time) > config.CIGARETTE_COOLDOWN_SEC:
            self.last_alert_time = now
            self.alert_count += 1
            alert_fired = True
            log_alert(
                event_type="cigarette_detected",
                details={
                    "confidence": round(self.last_confidence, 3),
                    "frames_confirmed": f"{sum(self.confirm_buffer)}/{config.CIGARETTE_CONFIRM_WINDOW}",
                },
            )

        return cigarette_detected, self.last_confidence, alert_fired

    def _is_confirmed(self) -> bool:
        return (
            len(self.confirm_buffer) == config.CIGARETTE_CONFIRM_WINDOW
            and sum(self.confirm_buffer) >= config.CIGARETTE_CONFIRM_FRAMES
        )