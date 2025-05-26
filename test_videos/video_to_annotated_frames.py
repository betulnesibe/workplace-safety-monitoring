import cv2
import os
from ultralytics import YOLO

# Paths
video_path = "test_videos/test2.mov"
output_dir = "outputs/test2_video"
os.makedirs(output_dir, exist_ok=True)

# Load model
model = YOLO("outputs/yolo_training/weights/best.pt")

# Read video
cap = cv2.VideoCapture(video_path)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_file = os.path.join(output_dir, f"frame_{frame_id:04}.jpg")
    cv2.imwrite(frame_file, frame)

    # Inference and save annotated frame
    results = model(frame, save=False)
    annotated_frame = results[0].plot()
    cv2.imwrite(os.path.join(output_dir, f"annotated_{frame_id:04}.jpg"), annotated_frame)
    frame_id += 1

cap.release()
print(f"✅ Done: {frame_id} frames processed.")