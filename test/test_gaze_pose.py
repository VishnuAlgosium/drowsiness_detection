"""
test_gaze_pose.py
------------------
Standalone visual test for gaze.py's head pose estimation. Draws a 3D axis
gizmo on the nose using MediaPipe's own facial transformation matrix, plus
live yaw/pitch/roll numbers, so you can visually confirm the pose matches
what you're actually doing with your head before trusting the distraction
thresholds in config.py.

Run from the same folder as gaze.py / config.py:
    python test_gaze_pose.py

Keys: c = capture current pose as baseline ("forward"), r = reset baseline
      to zero, q = quit
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
from gaze import head_pose_angles

CAM_INDEX = 0  # change if your camera isn't index 0

AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue


def draw_pose_axis(frame, rotation_matrix, origin, length=80):
    """Weak-perspective axis gizmo: rotate unit X/Y/Z vectors and draw them
    from the nose tip, ignoring true depth/camera intrinsics -- enough to
    visually confirm the pose direction without needing a calibrated camera."""
    axes = np.eye(3, dtype=np.float64)
    rotated = rotation_matrix @ axes

    for i, color in enumerate(AXIS_COLORS):
        dx, dy = rotated[0, i], rotated[1, i]
        end = (int(origin[0] + dx * length), int(origin[1] - dy * length))
        cv2.line(frame, origin, end, color, 2)


def main():
    base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    face_mesh = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    timestamp_ms = 0
    baseline_yaw, baseline_pitch, baseline_roll = 0.0, 0.0, 0.0

    print("[INFO] Press 'c' to capture current pose as baseline, 'r' to reset, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms += 33
        results = face_mesh.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks and results.facial_transformation_matrixes:
            landmarks = results.face_landmarks[0]
            transform = results.facial_transformation_matrixes[0]

            nose_x, nose_y = int(landmarks[config.NOSE_TIP_IDX].x * w), int(landmarks[config.NOSE_TIP_IDX].y * h)
            cv2.circle(frame, (nose_x, nose_y), 4, (0, 255, 255), -1)

            angles = head_pose_angles(transform)
            if angles:
                yaw, pitch, roll = angles
                rotation_matrix = np.array(transform)[:3, :3]
                draw_pose_axis(frame, rotation_matrix, (nose_x, nose_y))

                dev_yaw = yaw - baseline_yaw
                dev_pitch = pitch - baseline_pitch
                dev_roll = roll - baseline_roll

                looking_away = (
                    abs(dev_yaw) > config.YAW_ANGLE_MAX
                    or abs(dev_pitch) > config.PITCH_ANGLE_MAX
                    or abs(dev_roll) > config.ROLL_ANGLE_MAX
                )
                color = (0, 0, 255) if looking_away else (0, 255, 0)
                cv2.putText(frame, f"Yaw:{yaw:.1f}  Pitch:{pitch:.1f}  Roll:{roll:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Baseline yaw:{baseline_yaw:.1f} pitch:{baseline_pitch:.1f} roll:{baseline_roll:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, "LOOKING AWAY" if looking_away else "FORWARD",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Gaze Pose Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and results.face_landmarks and results.facial_transformation_matrixes:
            angles = head_pose_angles(results.facial_transformation_matrixes[0])
            if angles:
                baseline_yaw, baseline_pitch, baseline_roll = angles
                print(f"[INFO] Baseline captured: yaw={baseline_yaw:.1f} pitch={baseline_pitch:.1f} roll={baseline_roll:.1f}")
        elif key == ord("r"):
            baseline_yaw, baseline_pitch, baseline_roll = 0.0, 0.0, 0.0
            print("[INFO] Baseline reset to zero")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()