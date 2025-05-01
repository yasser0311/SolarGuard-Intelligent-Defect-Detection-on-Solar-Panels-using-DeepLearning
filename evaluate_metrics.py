from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "D:\\Projects\\Guvi_Project5\\Intelligaurd\\runs\\detect\\solar_guard_detector3\\weights\\best.pt"
DATASET_YAML = "D:\\Projects\\Guvi_Project5\\Intelligaurd\\data\\final_yolo_dataset\\data.yaml"

# === Load the trained model ===
model = YOLO(MODEL_PATH)

# === Evaluate on test set (specified by YAML under 'test') ===
metrics = model.val(
    data=DATASET_YAML,
    split="test",   # use test set
    conf=0.25,      # confidence threshold for predictions
    iou=0.6,        # IoU threshold for matching predictions
    save_json=True, # optional: save COCO format results
    plots=True      # generate PR curves, confusion matrix, etc.
)

# === Print summary ===
print(f"\n📊 mAP@0.5: {metrics.box.map50:.3f}")
print(f"📊 mAP@0.5:0.95: {metrics.box.map:.3f}")
print(f"📊 Precision: {metrics.box.precision:.3f}")
print(f"📊 Recall: {metrics.box.recall:.3f}")
