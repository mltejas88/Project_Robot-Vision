# from ultralytics import YOLO

# # Load your trained pose model
# model = YOLO("runs/pose/train/weights/best.pt")  # adjust train folder if needed

# # Run inference
# results = model(
#     "/home/tejas/YOLO/Extra_data/",   # path to image
#     conf=0.8,     # based on your F1/PR analysis
#     # iou=0.5,
#     show=False,    # display result
#     save=True     # save output image
# )


import cv2
from ultralytics import YOLO

model = YOLO("runs/pose/train/weights/best.pt")

cap = cv2.VideoCapture(0)  # webcam

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        frame,
        conf=0.8,
        verbose=False
    )

    r = results[0]

    # Draw results
    annotated = r.plot()

    cv2.imshow("YOLOv11 Pose - Live", annotated)

    # Access keypoints if needed
    if r.keypoints is not None:
        kpts = r.keypoints.xy.cpu().numpy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

