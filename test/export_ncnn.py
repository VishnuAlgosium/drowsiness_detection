from ultralytics import YOLO

# Load YOLO model
model = YOLO("../models/phone_detection_v6.pt")   # or yolov8n.pt, etc.

# Export directly to NCNN
model.export(format="ncnn")