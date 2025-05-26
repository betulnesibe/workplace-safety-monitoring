import os
import glob
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

def read_yolo_labels(folder, img_wh):
    labels = {}
    for label_file in glob.glob(os.path.join(folder, "*.txt")):
        name = os.path.basename(label_file)
        with open(label_file, "r") as f:
            boxes = []
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5]) 
                x1 = (x_center - w / 2) * img_wh[0]
                y1 = (y_center - h / 2) * img_wh[1]
                x2 = (x_center + w / 2) * img_wh[0]
                y2 = (y_center + h / 2) * img_wh[1]
                boxes.append([cls, x1, y1, x2, y2])
            labels[name] = boxes
    return labels

def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5):
    gt_matched = set()
    tp = 0
    for pred in pred_boxes:
        pred_cls, pred_box = pred[0], pred[1:]
        matched = False
        for i, gt in enumerate(gt_boxes):
            gt_cls, gt_box = gt[0], gt[1:]
            if i not in gt_matched and gt_cls == pred_cls:
                if iou(pred_box, gt_box) >= iou_threshold:
                    gt_matched.add(i)
                    matched = True
                    break
        tp += 1 if matched else 0
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(gt_matched)
    return tp, fp, fn

def evaluate(gt_dir, pred_dir, img_wh=(640, 640)):
    gt_labels = read_yolo_labels(gt_dir, img_wh)
    pred_labels = read_yolo_labels(pred_dir, img_wh)

    total_tp = total_fp = total_fn = 0
    for fname in gt_labels:
        gt = gt_labels.get(fname, [])
        pred = pred_labels.get(fname, [])
        tp, fp, fn = match_predictions(gt, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print(f"📊 Precision: {precision:.4f}")
    print(f"📊 Recall:    {recall:.4f}")
    print(f"📊 F1 Score:  {f1:.4f}")
    print(f"✅ Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

if __name__ == "__main__":
    evaluate("model/dataset/labels/test", "ensemble_labels_v3", img_wh=(640, 640))