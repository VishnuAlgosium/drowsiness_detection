"""
drowsy_detect
=============
Standalone webcam-based driver monitoring system.

Runs nine signals in one shared capture loop:
  - MediaPipe FaceLandmarker: drowsiness (EAR), eye occlusion, PERCLOS,
    yawning (MAR), gaze distraction, head drop
  - YOLO NCNN models: phone use, cigarette use, missing seatbelt

Each raises its own audio alert and is logged to drowsy_detect/../logs.
"""

__version__ = "2.0.0"