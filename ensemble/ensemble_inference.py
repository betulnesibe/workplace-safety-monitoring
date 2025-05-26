import os
import torch
import numpy as np
import cv2
from glob import glob
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pathlib import Path

# CONFIGURATION
model_paths = [
    "../runs/detect/crossval/fold_0_run/weights/best.pt",
    "../runs/detect/crossval/fold_1_run/weights/best.pt",
    "../runs/detect/crossval/fold_2_run/weights/best.pt",
    "../runs/detect/crossval/fold_3_run/weights/best.pt",
    "../runs/detect/crossval/fold_4_run/weights/best.pt"
]
image_folder = "../model/dataset/images/test"
output_folder = "ensemble_results_v3"
iou_thr = 0.55
skip_box_thr = 0.001
weights = [1, 1, 1, 1, 1]  # You can tune these later

os.makedirs(output_folder, exist_ok=True)

# Load all models
models = [YOLO(p) for p in model_paths]

# Load test images
image_paths = sorted(glob(f"{image_folder}/*.jpg"))

for img_path in image_paths:
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    img_name = Path(img_path).stem

    boxes_list, scores_list, labels_list = [], [], []

    for model in models:
        results = model(img_path)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)

        # Convert boxes from [x1, y1, x2, y2] to normalized [x, y, x, y]
        norm_boxes = boxes.copy()
        norm_boxes[:, [0, 2]] /= width
        norm_boxes[:, [1, 3]] /= height
        norm_boxes = norm_boxes.tolist()

        boxes_list.append(norm_boxes)
        scores_list.append(scores.tolist())
        labels_list.append(labels.tolist())

    # Weighted Boxes Fusion
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )

    # Convert boxes back to original scale
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, f'{model.names[int(label)]} {score:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imwrite(f"{output_folder}/{img_name}.jpg", image)
    label_output_dir = "ensemble_labels_v3"
    os.makedirs(label_output_dir, exist_ok=True)

    label_path = os.path.join(label_output_dir, f"{img_name}.txt")
    with open(label_path, "w") as f:
        for box, score, label in zip(boxes, scores, labels):
            # Normalize coordinates
            x_center = ((box[0] + box[2]) / 2) / width
            y_center = ((box[1] + box[3]) / 2) / height
            box_width = (box[2] - box[0]) / width
            box_height = (box[3] - box[1]) / height
            f.write(f"{int(label)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {score:.6f}\n")
print("✅ Ensembling completed. Images saved to:", output_folder)