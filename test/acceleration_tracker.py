"""
Acceleration Tracking Simulator with FastAPI
--------------------------------------------
Simulates a vehicle that automatically speeds up to ~60 km/h and slows down
to ~10 km/h, again and again, like city driving. Data is served through
FastAPI for a driving management system.

Install:  pip install fastapi uvicorn
Run:      python acceleration_tracker.py
Docs:     http://localhost:8000/docs

Endpoints:
  GET  /telemetry/latest           -> newest reading
  GET  /telemetry/history?limit=50 -> last N readings
  GET  /events?limit=20            -> harsh acceleration / harsh braking events
  GET  /summary                    -> trip statistics
  WS   /ws/telemetry               -> live stream (10 readings per second)
"""
import asyncio, random, uuid
from collections import deque
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ---------------- Settings ----------------
HZ = 10                    # readings per second
HIGH_SPEED = (55, 60)      # km/h range to accelerate up to
LOW_SPEED = (0, 15)       # km/h range to slow down to
CRUISE_TIME = (2, 5)       # seconds to hold speed before changing
HARSH_ACCEL_G = 0.30       # harsh acceleration threshold
HARSH_BRAKE_G = -0.40      # harsh braking threshold
HARSH_CHANCE = 0.25        # chance a phase is driven aggressively
G = 9.81

# ---------------- Shared state ----------------
history = deque(maxlen=3000)   # about 5 minutes of data
events = deque(maxlen=500)
trip = {"trip_id": str(uuid.uuid4())[:8], "vehicle_id": "SIM-001",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "distance_km": 0.0, "max_speed_kmh": 0.0,
        "harsh_accel_count": 0, "harsh_brake_count": 0}


async def simulate():
    """Runs forever in the background and produces readings."""
    dt = 1 / HZ
    speed = 0.0      # m/s
    accel = 0.0      # m/s^2
    phase, target, hold, rate = "accelerating", random.uniform(*HIGH_SPEED) / 3.6, 0.0, 2.0
    harsh_timer = 0.0

    while True:
        # ---- decide the wanted acceleration for this phase ----
        if phase == "accelerating":
            wanted = rate if speed < target else 0.0
            if speed >= target:
                phase, hold = "cruising_high", random.uniform(*CRUISE_TIME)
        elif phase == "decelerating":
            wanted = -rate if speed > target else 0.0
            if speed <= target:
                phase, hold = "cruising_low", random.uniform(*CRUISE_TIME)
        else:  # cruising
            wanted = random.gauss(0, 0.1)
            hold -= dt
            if hold <= 0:
                aggressive = random.random() < HARSH_CHANCE
                if phase == "cruising_high":
                    phase, target = "decelerating", random.uniform(*LOW_SPEED) / 3.6
                    rate = random.uniform(4.5, 6.0) if aggressive else random.uniform(1.0, 2.5)
                else:
                    phase, target = "accelerating", random.uniform(*HIGH_SPEED) / 3.6
                    rate = random.uniform(3.5, 4.5) if aggressive else random.uniform(1.0, 2.2)

        # smooth change of acceleration (a real car can't jump instantly)
        max_jerk = 8.0 * dt
        accel += max(-max_jerk, min(max_jerk, wanted - accel))
        speed = max(0.0, speed + accel * dt)

        # ---- measured values (with small sensor noise) ----
        accel_meas = accel + random.gauss(0, 0.05)
        g_force = accel_meas / G
        speed_kmh = speed * 3.6

        # harsh event detection: must last at least 0.5 s
        if g_force >= HARSH_ACCEL_G or g_force <= HARSH_BRAKE_G:
            harsh_timer += dt
        else:
            harsh_timer = 0.0
        event = None
        if abs(harsh_timer - 0.5) < dt / 2:          # fire once per event
            event = "harsh_acceleration" if g_force > 0 else "harsh_braking"
            trip["harsh_accel_count" if g_force > 0 else "harsh_brake_count"] += 1

        trip["distance_km"] += speed * dt / 1000
        trip["max_speed_kmh"] = max(trip["max_speed_kmh"], speed_kmh)

        reading = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "vehicle_id": trip["vehicle_id"],
            "trip_id": trip["trip_id"],
            "speed_kmh": round(speed_kmh, 2),
            "acceleration_mps2": round(accel_meas, 3),
            "acceleration_g": round(g_force, 3),
            "state": phase,
            "target_speed_kmh": round(target * 3.6, 1),
            "event": event,
        }
        history.append(reading)
        if event:
            events.append(reading)

        await asyncio.sleep(dt)


# ---------------- FastAPI ----------------
app = FastAPI(title="Acceleration Tracking Simulator")


@app.on_event("startup")
async def start_simulation():
    asyncio.create_task(simulate())


@app.get("/telemetry/latest")
def latest():
    return history[-1] if history else {}


@app.get("/telemetry/history")
def get_history(limit: int = 50):
    return list(history)[-limit:]


@app.get("/events")
def get_events(limit: int = 20):
    return list(events)[-limit:]


@app.get("/summary")
def summary():
    data = dict(trip)
    data["distance_km"] = round(data["distance_km"], 3)
    data["max_speed_kmh"] = round(data["max_speed_kmh"], 1)
    data["current_speed_kmh"] = history[-1]["speed_kmh"] if history else 0
    return data


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    last = None
    try:
        while True:
            if history and history[-1] is not last:
                last = history[-1]
                await ws.send_json(last)
            await asyncio.sleep(1 / HZ)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)