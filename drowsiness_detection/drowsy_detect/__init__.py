"""
drowsy_detect
=============
Standalone webcam-based driver monitoring system.

Combines three signals in one shared capture loop:
  - Drowsiness: MediaPipe FaceLandmarker + EAR (Eye Aspect Ratio)
  - Yawning:    MediaPipe FaceLandmarker + MAR (Mouth Aspect Ratio)
  - Phone use:  YOLO ONNX object detection

Each raises its own audio alert and is logged to drowsy_detect/../logs.
"""

__version__ = "2.0.0"
