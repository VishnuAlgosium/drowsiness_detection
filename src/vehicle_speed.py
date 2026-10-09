"""
vehicle_speed.py
----------------
Reads live vehicle speed from the driving-management telemetry WebSocket
(ws://<server>:8000/ws/telemetry) in a background thread, and decides whether
driver monitoring should run (only while the vehicle is moving).

Install:  pip install websockets

Config values (all optional, defaults shown):
    DMS_SPEED_GATE_ENABLED     = True    # False = always monitor, ignore speed
    TELEMETRY_WS_URL           = "ws://localhost:8000/ws/telemetry"
    DMS_MIN_SPEED_KMH          = 10.0    # monitor only above this speed
    DMS_SPEED_OFF_DELAY_SEC    = 3.0     # stay ON this long after dropping below
    TELEMETRY_STALE_SEC        = 3.0     # no data for this long = speed unknown
    DMS_RUN_WHEN_SPEED_UNKNOWN = True    # safety: keep monitoring if telemetry is lost
    TELEMETRY_RECONNECT_SEC    = 2.0
"""

import json
import math
import threading
import time

from . import config

try:
    from websockets.sync.client import connect
except ImportError:          # websockets not installed (or older than v12)
    connect = None


class VehicleSpeedClient:
    """Keeps the latest speed from the telemetry WebSocket. Reconnects forever."""

    def __init__(self, url=None):
        self.url = url or getattr(config, "TELEMETRY_WS_URL", "ws://localhost:8000/ws/telemetry")
        self._retry_sec = getattr(config, "TELEMETRY_RECONNECT_SEC", 2.0)
        self._lock = threading.Lock()
        self._speed = None
        self._accel_g = None
        self._updated = -math.inf
        self.connected = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vehicle-speed", daemon=True)

    def start(self) -> None:
        if not getattr(config, "DMS_SPEED_GATE_ENABLED", True):
            return
        if connect is None:
            print("[ERROR] 'websockets' package missing (pip install websockets) -- "
                  "vehicle speed unavailable")
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self):
        """Return (speed_kmh or None, accel_g or None, age_sec)."""
        with self._lock:
            return self._speed, self._accel_g, time.monotonic() - self._updated

    def _run(self) -> None:
        last_error_print = -math.inf
        while not self._stop.is_set():
            try:
                with connect(self.url, open_timeout=5, close_timeout=1) as ws:
                    self.connected = True
                    print(f"[INFO] Telemetry connected: {self.url}")
                    while not self._stop.is_set():
                        try:
                            message = ws.recv(timeout=1.0)
                        except TimeoutError:
                            continue
                        data = json.loads(message)
                        with self._lock:
                            self._speed = float(data.get("speed_kmh", 0.0))
                            self._accel_g = data.get("acceleration_g")
                            self._updated = time.monotonic()
            except Exception as exc:
                if self.connected:
                    print(f"[WARN] Telemetry disconnected ({exc}) -- reconnecting")
                elif time.monotonic() - last_error_print > 30:
                    print(f"[WARN] Telemetry not reachable at {self.url} ({exc}) -- retrying")
                    last_error_print = time.monotonic()
                self.connected = False
                self._stop.wait(self._retry_sec)


class SpeedGate:
    """Turns driver monitoring ON when speed > DMS_MIN_SPEED_KMH, and OFF only
    after it stays at/below that for DMS_SPEED_OFF_DELAY_SEC (stops flicker at
    traffic-light crawl speeds)."""

    def __init__(self, client: VehicleSpeedClient):
        self.client = client
        self.enabled = getattr(config, "DMS_SPEED_GATE_ENABLED", True)
        self.min_speed = getattr(config, "DMS_MIN_SPEED_KMH", 10.0)
        self.off_delay = getattr(config, "DMS_SPEED_OFF_DELAY_SEC", 3.0)
        self.stale_sec = getattr(config, "TELEMETRY_STALE_SEC", 3.0)
        self.run_when_unknown = getattr(config, "DMS_RUN_WHEN_SPEED_UNKNOWN", True)
        self.active = not self.enabled
        self.speed = None
        self.reason = "gate disabled" if not self.enabled else "waiting for telemetry"
        self._below_since = None

    @property
    def status_text(self) -> str:
        if self.speed is None:
            return self.reason
        return f"{self.speed:.1f} km/h"

    def update(self, now: float) -> bool:
        if not self.enabled:
            return True

        speed, _, age = self.client.snapshot()
        if speed is None or age > self.stale_sec:
            self.speed = None
            self.reason = "speed unknown (no telemetry)"
            self._below_since = None
            self.active = self.run_when_unknown
            return self.active

        self.speed = round(speed, 1)
        if speed > self.min_speed:
            self._below_since = None
            self.active = True
            self.reason = "moving"
        elif self.active:
            if self._below_since is None:
                self._below_since = now
            if now - self._below_since >= self.off_delay:
                self.active = False
                self.reason = "slow / stopped"
        else:
            self.reason = "slow / stopped"
        return self.active