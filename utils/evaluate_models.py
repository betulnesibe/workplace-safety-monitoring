# evaluate_models.py

from ultralytics import YOLO
import pandas as pd
import numpy as np
from pathlib import Path


def safe_float(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return float(x.item())  # scalar (0D)
        elif x.ndim == 1 and x.size > 0:
            return float(np.mean(x))  # take mean of per-class stats
        else:
            raise ValueError(f"Unexpected array shape: {x.shape}")
    return float(x)
# List of models to evaluate
model_paths = {
    "Base": "outputs/yolo_training/weights/best.pt",
    "Fine-tuned": "outputs/yolov8n-tuned2/weights/best.pt",
    "Fold 0": "runs/detect/crossval/fold_0_run/weights/best.pt",
    "Fold 1": "runs/detect/crossval/fold_1_run/weights/best.pt",
    "Fold 2": "runs/detect/crossval/fold_2_run/weights/best.pt",
    "Fold 3": "runs/detect/crossval/fold_3_run/weights/best.pt",
    "Fold 4": "runs/detect/crossval/fold_4_run/weights/best.pt",
    "Ensemble-trained": "runs/detect/distillation_ensemble/weights/best.pt"
}

data_yaml = "model/dataset/data.yaml"

overall_metrics = []
label_metrics = []

for name, model_path in model_paths.items():
    print(f"\n🔍 Evaluating {name}...")
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split="val", verbose=False)

    # Store overall metrics
    overall_metrics.append({
        "Model": name,
        "Precision": float(results.box.p.mean()),
        "Recall": float(results.box.r.mean()),
        "mAP@0.5": float(results.box.map50.mean()),
        "mAP@0.5:0.95": float(results.box.map.mean())
    })

    # Try per-class metrics
    try:
        if hasattr(results.box.map50, "__len__") and len(results.box.map50) == len(model.names):
            for i, cls_name in model.names.items():
                label_metrics.append({
                    "Model": name,
                    "Class": "all",
                    "Precision": safe_float(results.box.p),
                    "Recall": safe_float(results.box.r),
                    "mAP@0.5": safe_float(results.box.map50),
                    "mAP@0.5:0.95": safe_float(results.box.map)
                })
        else:
            raise ValueError("Per-class stats not available")
    except Exception as e:
        print(f"⚠️  Could not get per-class results for {name}: {e}")
        label_metrics.append({
            "Model": name,
            "Class": "all",
            "Precision": safe_float(results.box.p),
            "Recall": safe_float(results.box.r),
            "mAP@0.5": safe_float(results.box.map50),
            "mAP@0.5:0.95": safe_float(results.box.map)
        })

# Save CSVs
pd.DataFrame(overall_metrics).to_csv("model_overall_results.csv", index=False)
pd.DataFrame(label_metrics).to_csv("model_per_class_results.csv", index=False)

print("\n✅ Evaluation complete. Results saved to:")
print(" - model_overall_results.csv")
print(" - model_per_class_results.csv")