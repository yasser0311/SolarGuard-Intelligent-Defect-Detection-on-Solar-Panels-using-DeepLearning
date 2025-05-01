from ultralytics import YOLO
import os
import cv2
from pathlib import Path

# === CONFIG ===
MODEL_PATH = "D:\\Projects\\Guvi_Project5\\Intelligaurd\\runs\\detect\\solar_guard_detector3\\weights\\best.pt"
TEST_IMAGES_DIR = "D:\\Projects\\Guvi_Project5\\Intelligaurd\\data\\final_yolo_dataset\\test\\images"
OUTPUT_DIR = "runs/detect/solar_guard_detector2/test_predictions"
CONF_THRESHOLD = 0.3  # confidence threshold for predictions

# === Load model ===
model = YOLO(MODEL_PATH)

# === Run inference on test images ===
results = model.predict(
    source=TEST_IMAGES_DIR,
    save=True,
    save_txt=False,
    save_crop=False,
    project="runs/detect",
    name="solar_guard_detector2/test_predictions",
    conf=CONF_THRESHOLD
)

print(f"Inference completed. Results saved to: {OUTPUT_DIR}")
