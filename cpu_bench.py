#!/usr/bin/env python
"""
Standalone CPU-only benchmark for the face-mesh pipeline.
Forces CPU execution, hides the GPU from the process, pins to 4 threads
(simulating a 4-core device like Raspberry Pi 5), and prints FPS stats.

Run:  python cpu_bench.py [seconds]
"""

import os
import sys
import time

# ── Force CPU-only BEFORE importing cv2 / mediapipe ─────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # hide GPU from any CUDA-aware lib
os.environ["OMP_NUM_THREADS"] = "4"           # cap OpenMP threads (4-core sim)
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

import cv2
import mediapipe as mp
import numpy as np

cv2.ocl.setUseOpenCL(False)   # no OpenCL path in OpenCV
cv2.setNumThreads(4)          # cap OpenCV's internal thread pool to 4

# Optional: try to pin this whole process to 4 CPU cores (Linux only)
try:
    os.sched_setaffinity(0, {0, 1, 2, 3})
    print("[INFO] Process pinned to CPU cores 0-3")
except (AttributeError, OSError):
    print("[INFO] CPU affinity pinning not available on this OS/venv")

TEST_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

print(f"[INFO] cv2.ocl useOpenCL = {cv2.ocl.useOpenCL()}")   # should print False
print(f"[INFO] cv2 threads       = {cv2.getNumThreads()}")   # should print 4
print(f"[INFO] Running benchmark for {TEST_SECONDS}s...\n")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

fps_samples = []
start = time.time()
prev = start
frame_count = 0

try:
    while time.time() - start < TEST_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps_samples.append(1.0 / dt)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        _ = face_mesh.process(rgb)   # the actual CPU-bound inference
        frame_count += 1

finally:
    cap.release()
    face_mesh.close()

fps_arr = np.array(fps_samples) if fps_samples else np.array([0])
print("\n──────── CPU BENCHMARK RESULTS ────────")
print(f"Frames processed : {frame_count}")
print(f"Avg FPS          : {fps_arr.mean():.2f}")
print(f"Min FPS          : {fps_arr.min():.2f}")
print(f"Max FPS          : {fps_arr.max():.2f}")
print(f"----------------------------------------")
print("While this ran, check `nvidia-smi` GPU-Util — it should stay flat,")
print("not spike, confirming no GPU compute was used.")