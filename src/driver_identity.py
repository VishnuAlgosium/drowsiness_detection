"""
driver_identity.py
------------------
Who is driving, decided robustly from FaceIdentifier results.

A single frame's match is not trusted:
  - IDENTIFYING: when a face appears, sample DRIVER_ID_VOTES good frames
    (one every DRIVER_ID_SAMPLE_EVERY_SEC) and decide by majority: a known
    employee if enough votes agree on the same ID, else UNKNOWN.
  - CONFIRMED: re-check one frame every DRIVER_ID_REVERIFY_SEC. Only
    DRIVER_ID_REVERIFY_MISMATCHES disagreeing checks IN A ROW send it back
    to IDENTIFYING, so one bad frame never changes the driver.
  - Face gone for DRIVER_ID_RESET_AFTER_FACE_LOSS_SEC -> IDENTIFYING again
    (someone else may have sat down).

update() returns events for detector.py to log / alert on:
  ("driver_identified", info)  first decision of the session
  ("unknown_driver", info)     decision = not an enrolled employee
  ("driver_changed", info)     decision differs from the previous driver

DriverProfiles stores each known driver's calibrated EAR/pose baseline, so a
returning driver gets their own thresholds immediately.
"""

import json
import os
from collections import Counter
from dataclasses import asdict

from . import config
from .calibration import Baseline


def _cfg(name, default):
    return getattr(config, name, default)


UNKNOWN = "UNKNOWN"


class DriverIdentityTracker:
    def __init__(self):
        self.votes_needed = _cfg("DRIVER_ID_VOTES", 5)
        self.min_agreement = _cfg("DRIVER_ID_MIN_AGREEMENT", 0.6)
        self.sample_every = _cfg("DRIVER_ID_SAMPLE_EVERY_SEC", 0.2)
        self.reverify_every = _cfg("DRIVER_ID_REVERIFY_SEC", 30.0)
        self.reverify_mismatches = _cfg("DRIVER_ID_REVERIFY_MISMATCHES", 3)
        self.reset_after_loss = _cfg("DRIVER_ID_RESET_AFTER_FACE_LOSS_SEC", 10.0)

        self.phase = "identifying"
        self.driver_id = None          # employee ID, UNKNOWN, or None (not decided yet)
        self.driver_name = ""
        self.score = 0.0
        self._votes = []               # (id_or_UNKNOWN, name, score)
        self._last_sample_t = -1e9
        self._mismatch_run = 0
        self._ever_decided = False

    # ── what the detector asks before doing any work ──
    def wants_sample(self, now: float) -> bool:
        gap = self.sample_every if self.phase == "identifying" else self.reverify_every
        return now - self._last_sample_t >= gap

    @property
    def label(self) -> str:
        if self.driver_id is None:
            return "identifying..."
        return UNKNOWN if self.driver_id == UNKNOWN else f"{self.driver_id} {self.driver_name}"

    def context(self) -> dict:
        """Fields attached to every logged alert."""
        return {"driver_id": self.driver_id or "", "driver_name": self.driver_name if self.driver_id != UNKNOWN else ""}

    # ── inputs ──
    def face_lost(self, face_lost_sec: float) -> None:
        if self.phase == "confirmed" and face_lost_sec >= self.reset_after_loss:
            self.phase = "identifying"     # keep driver_id to compare against the next decision
            self._votes.clear()
            self._mismatch_run = 0
            self._last_sample_t = -1e9

    def update(self, now: float, status: str, emp_id: str, emp_name: str, score: float):
        """Feed one identification result (from a quality-checked frame).
        Returns a list of (event_type, info) tuples (usually empty)."""
        self._last_sample_t = now
        vote = emp_id if status == "KNOWN" else UNKNOWN
        events = []

        if self.phase == "confirmed":
            if vote == self.driver_id:
                self._mismatch_run = 0
                self.score = score if vote != UNKNOWN else self.score
            else:
                self._mismatch_run += 1
                if self._mismatch_run >= self.reverify_mismatches:
                    self.phase = "identifying"
                    self._votes.clear()
                    self._mismatch_run = 0
            return events

        self._votes.append((vote, emp_name, score))
        if len(self._votes) > self.votes_needed:
            self._votes.pop(0)
        if len(self._votes) < self.votes_needed:
            return events

        winner, n = Counter(v for v, _, _ in self._votes).most_common(1)[0]
        if n / len(self._votes) < self.min_agreement:
            return events                     # no clear majority yet: keep sampling (window slides)
        if winner != UNKNOWN:
            scores = [s for v, _, s in self._votes if v == winner]
            name = next(nm for v, nm, _ in self._votes if v == winner)
            new_score = sum(scores) / len(scores)
        else:
            name, new_score = "", max(s for _, _, s in self._votes)

        previous = self.driver_id
        self.driver_id, self.driver_name, self.score = winner, name, new_score
        self.phase = "confirmed"
        self._votes.clear()
        self._mismatch_run = 0

        info = {"driver_id": winner, "driver_name": name, "score": round(new_score, 3),
                "votes": f"{n}/{self.votes_needed}"}
        if not self._ever_decided:
            self._ever_decided = True
            events.append(("driver_identified", info))
        elif previous != winner:
            events.append(("driver_changed", dict(info, previous_driver_id=previous)))
        if winner == UNKNOWN and previous != UNKNOWN:
            events.append(("unknown_driver", info))
        return events


class DriverProfiles:
    """Per-driver calibrated baselines, persisted as JSON."""

    def __init__(self, path=None):
        self.path = path or _cfg("DRIVER_PROFILE_PATH", os.path.join(config.LOG_DIR, "driver_profiles.json"))
        self._data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except Exception as e:
                print(f"[WARN] Could not read driver profiles {self.path}: {e}")

    def get(self, driver_id):
        d = self._data.get(driver_id)
        return Baseline(**d) if d else None

    def save(self, driver_id, baseline: Baseline) -> None:
        if not driver_id or driver_id == UNKNOWN or not baseline.calibrated:
            return
        self._data[driver_id] = asdict(baseline)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)