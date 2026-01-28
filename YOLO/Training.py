from ultralytics import YOLO

# ----------------------------
# Load pretrained YOLOv11 pose model
# ----------------------------
model = YOLO("yolo11m-pose.pt")   # or yolo11s-pose.pt, yolo11m-pose.pt

# ----------------------------
# Fine-tune the model
# ----------------------------
results = model.train(
    data="config.yaml",     # path to dataset config
    epochs=150,           # fine-tuning epochs
    imgsz=640,            # image size
    batch=4,             # adjust to GPU memory
    device=0,             # 0 = GPU, "cpu" for CPU
    #pretrained=True,      # IMPORTANT: fine-tuning
)
