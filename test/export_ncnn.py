from ultralytics import YOLO

# Load YOLO model
model = YOLO("../models/cigarette_detection_v3.pt")   # or yolov8n.pt, etc.

# Export directly to NCNN
model.export(format="ncnn")