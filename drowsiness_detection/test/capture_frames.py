"""
capture_frames.py
------------------
Grabs a frame every N seconds from a video source and saves it
inside a folder named after the person.

Usage:
    python capture_frames.py
"""

import os
import time
import cv2

SOURCE = 0                    # 0/1/2 for webcam, or video/RTSP path
INTERVAL_SEC = 0.5              # seconds between saved frames
OUTPUT_DIR = "captured_frames"


# ---------------------------------------------------------
# Ask for person's name
# ---------------------------------------------------------
person_name = input("Enter person's name: ").strip()

if not person_name:
    raise ValueError("Person name cannot be empty.")

# Replace spaces with underscores for folder/file names
person_name = person_name.replace(" ", "_")

# Create person's folder
person_dir = os.path.join(OUTPUT_DIR, person_name)
os.makedirs(person_dir, exist_ok=True)


# ---------------------------------------------------------
# Open camera/video
# ---------------------------------------------------------
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Could not open source: {SOURCE}")


frame_num = 0
saved_count = 0
last_save_time = 0.0

print()
print(f"[INFO] Person      : {person_name}")
print(f"[INFO] Saving every: {INTERVAL_SEC}s")
print(f"[INFO] Output      : '{person_dir}/'")
print("[INFO] Press 'q' or Ctrl+C to stop")
print()


try:
    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("[INFO] No more frames or camera not returning frames.")
            break

        frame_num += 1
        now = time.time()

        # Save frame every INTERVAL_SEC seconds
        if now - last_save_time >= INTERVAL_SEC:

            filename = os.path.join(
                person_dir,
                f"{person_name}_{saved_count:05d}.jpg"
            )

            success = cv2.imwrite(filename, frame)

            if success:
                saved_count += 1
                last_save_time = now
                print(f"[SAVED] {filename}")
            else:
                print(f"[ERROR] Could not save {filename}")

        # Display camera feed
        cv2.imshow("Frame Capture", frame)

        # Press q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Stopped by user")
            break


except KeyboardInterrupt:
    print("\n[INFO] Stopped by user")


finally:
    cap.release()
    cv2.destroyAllWindows()

    print()
    print(f"[INFO] Done.")
    print(f"[INFO] Person: {person_name}")
    print(f"[INFO] Frames saved: {saved_count}")
    print(f"[INFO] Saved in: {person_dir}")