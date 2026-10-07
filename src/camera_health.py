"""
camera_health.py
----------------
1. open_camera_with_retry(): keeps retrying until the camera is available AND
   actually delivers a frame (isOpened() alone can be True on a dead device).
2. CameraOcclusionMonitor: detects when the lens is FULLY or PARTLY covered
   (hand, finger, tape, cloth, sticker, heavy fog) from image statistics.

All thresholds are read from config with safe defaults, so nothing breaks
if you haven't added them to config.py yet.
"""

import time

import cv2
import numpy as np

from . import config


def _cfg(name, default):
    return getattr(config, name, default)


# ─────────────────────────────────────────────────────────────────────────────
# Camera open with retry
# ─────────────────────────────────────────────────────────────────────────────
def open_camera_with_retry(open_fn, should_abort=lambda: False, on_unavailable=None):
    """
    Calls open_fn() repeatedly until a camera that returns a real frame is found.

    open_fn         -- function returning a cv2.VideoCapture (e.g. detector._open_camera)
    should_abort    -- returns True to stop retrying (e.g. SIGTERM received)
    on_unavailable  -- called ONCE on the first failure (use it to alert/log)

    config.CAMERA_OPEN_RETRY_ATTEMPTS = 0 means retry forever.
    Returns an opened cv2.VideoCapture, or None if it gave up / was aborted.
    """
    max_attempts = _cfg("CAMERA_OPEN_RETRY_ATTEMPTS", 0)
    delay = _cfg("CAMERA_OPEN_RETRY_DELAY_SEC", 2.0)
    max_delay = _cfg("CAMERA_OPEN_RETRY_MAX_DELAY_SEC", 30.0)

    attempt = 0
    notified = False

    while not should_abort():
        attempt += 1
        cap = open_fn()

        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                if attempt > 1:
                    print(f"[INFO] Camera available again (attempt {attempt})")
                return cap
            print(f"[WARN] Camera opened but gave no frame (attempt {attempt})")
        else:
            print(f"[WARN] Camera not available (attempt {attempt})")

        cap.release()

        if not notified and on_unavailable is not None:
            on_unavailable()
            notified = True

        if max_attempts and attempt >= max_attempts:
            print(f"[ERROR] Camera still unavailable after {attempt} attempts")
            return None

        print(f"[INFO] Retrying camera in {delay:.1f}s ...")
        end = time.monotonic() + delay
        while time.monotonic() < end and not should_abort():
            time.sleep(0.1)
        delay = min(delay * 1.5, max_delay)   # gentle backoff

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Lens occlusion / camera blocked (full or partial)
# ─────────────────────────────────────────────────────────────────────────────
class CameraOcclusionMonitor:
    """
    FULL block  -- whole frame very dark, or almost featureless
                   (low contrast AND almost no edges).
    PARTIAL block -- the frame is split into a 4x4 grid of cells. A cell
                   counts as covered if it is very dark, or if it has lost
                   most of its texture compared with a REFERENCE taken while
                   the view was known to be clear (during calibration, then
                   slowly refreshed while driving). A finger or hand close to
                   the lens is blurry, so its cells lose their edges.
                   If >= CAMERA_PARTIAL_BLOCK_RATIO of the cells are covered
                   -> partially blocked.

    Cell texture is normalised by the frame's 75th-percentile cell texture,
    so an overall lighting change (tunnel, cloud) doesn't count as a block.

    Flags:
      suspect -- the LAST measured frame looked blocked (instant, no hold)
      blocked -- it has looked blocked for CAMERA_BLOCK_HOLD_SEC (confirmed);
                 clears only after CAMERA_BLOCK_CLEAR_SEC of clear frames.
    """

    ROWS, COLS = 4, 4
    SMALL_W, SMALL_H = 160, 120

    def __init__(self):
        self.dark_mean = _cfg("CAMERA_BLOCK_DARK_MEAN", 25.0)          # 0-255
        self.min_std = _cfg("CAMERA_BLOCK_MIN_STD", 12.0)              # contrast
        self.min_edges = _cfg("CAMERA_BLOCK_MIN_LAPLACIAN_VAR", 15.0)  # sharpness/edges
        self.partial_ratio = _cfg("CAMERA_PARTIAL_BLOCK_RATIO", 0.25)  # 4 of 16 cells
        self.lost_edge_frac = _cfg("CAMERA_PARTIAL_LOST_EDGE_FRAC", 0.2)
        self.ref_alpha = _cfg("CAMERA_REFERENCE_ALPHA", 0.02)
        self.hold_sec = _cfg("CAMERA_BLOCK_HOLD_SEC", 2.0)
        self.clear_sec = _cfg("CAMERA_BLOCK_CLEAR_SEC", 1.0)
        self.check_every = max(1, _cfg("CAMERA_BLOCK_CHECK_EVERY_N_FRAMES", 3))

        self._ref_rel = None
        self.reset()

    # ── measurement ──
    def _measure(self, frame):
        small = cv2.resize(frame, (self.SMALL_W, self.SMALL_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)

        R, C = self.ROWS, self.COLS
        ch, cw = self.SMALL_H // R, self.SMALL_W // C
        g = gray.astype(np.float32).reshape(R, ch, C, cw)
        l = lap.reshape(R, ch, C, cw)

        cell_mean = g.mean(axis=(1, 3))
        cell_edges = l.var(axis=(1, 3))
        return float(gray.mean()), float(gray.std()), float(lap.var()), cell_mean, cell_edges

    def update(self, frame, now) -> bool:
        """Returns True while the camera is considered blocked (fully or partly)."""
        self._frame_counter += 1
        if self._frame_counter % self.check_every != 0:
            return self.blocked
        self._last_measured = self._frame_counter

        self.mean, self.std, self.edges, cell_mean, cell_edges = self._measure(frame)

        too_dark = self.mean < self.dark_mean
        featureless = self.std < self.min_std and self.edges < self.min_edges

        norm = float(np.percentile(cell_edges, 75)) + 1e-6
        self._cell_rel = cell_edges / norm

        covered = cell_mean < self.dark_mean
        if self._ref_rel is not None:
            textured = self._ref_rel > 0.3       # only cells that had real texture when clear
            covered = covered | (textured & (self._cell_rel < self.lost_edge_frac * self._ref_rel))
        self._cell_covered = covered
        self.blocked_ratio = float(covered.mean())
        partial = self.blocked_ratio >= self.partial_ratio

        self.suspect = too_dark or featureless or partial
        if self.suspect:
            self.reason = ("too_dark" if too_dark else
                           "featureless" if featureless else
                           f"partially_blocked {self.blocked_ratio:.0%}")

        if self.suspect:
            self._clear_since = None
            if self._suspect_since is None:
                self._suspect_since = now
            if not self.blocked and now - self._suspect_since >= self.hold_sec:
                self.blocked = True
        else:
            self._suspect_since = None
            if self.blocked:
                if self._clear_since is None:
                    self._clear_since = now
                if now - self._clear_since >= self.clear_sec:
                    self.blocked = False
                    self.reason = ""
                    self._clear_since = None

        return self.blocked

    # ── clear-view reference for partial detection ──
    def update_reference(self, alpha=None) -> None:
        """Learn what the clear view looks like. Call only when the view is known
        to be good (face detected, not suspect). Uses this frame's measurement only."""
        if self._cell_rel is None or self._last_measured != self._frame_counter:
            return
        if self._ref_rel is None:
            self._ref_rel = self._cell_rel.copy()
            return
        if self._cell_covered is not None and self._cell_covered.any():
            return          # never learn a (possibly) covered view
        a = self.ref_alpha if alpha is None else alpha
        self._ref_rel = (1 - a) * self._ref_rel + a * self._cell_rel

    @property
    def has_reference(self) -> bool:
        return self._ref_rel is not None

    def reset(self, keep_reference: bool = False) -> None:
        self.blocked = False
        self.suspect = False
        self.reason = ""
        self.blocked_ratio = 0.0
        self.mean = self.std = self.edges = 0.0
        self._frame_counter = 0
        self._last_measured = -1
        self._suspect_since = None
        self._clear_since = None
        self._cell_rel = None
        self._cell_covered = None
        if not keep_reference:
            self._ref_rel = None