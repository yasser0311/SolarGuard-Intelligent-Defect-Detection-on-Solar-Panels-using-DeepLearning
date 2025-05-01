import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
CLASS_NAMES = ['Clean', 'Dusty', 'Bird-drop', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
OBJECT_CLASSES = ['Bird-drop', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
CNN_MODEL_PATH = "D:\\Projects\\Guvi_Project5\\Intelligaurd\\models\\mobilenetv2_panel_classifier.h5"
YOLO_MODEL_PATH = "D:\\Projects\\Guvi_Project5\\Intelligaurd\\runs\\detect\\solar_guard_detector3\\weights\\best.pt"

# === Load Models ===
@st.cache_resource
def load_models():
    cnn_model = load_model(CNN_MODEL_PATH)
    yolo_model = YOLO(YOLO_MODEL_PATH)
    return cnn_model, yolo_model

cnn_model, yolo_model = load_models()

# === Classification Prediction ===
def predict_class(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = cnn_model.predict(img_array)[0]
    class_idx = np.argmax(preds)
    return CLASS_NAMES[class_idx], preds[class_idx]

# === Object Detection Prediction ===
def predict_objects(image_path):
    results = yolo_model.predict(source=image_path, save=False, conf=0.3)
    return results[0]

# === UI ===
st.title("☀️SolarGuard⚡: Intelligent Defect Detection on Solar Panels🛡️")

st.sidebar.header("Upload Solar Panel Image")
image_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if image_file:
    img = Image.open(image_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.subheader("1. Panel Condition Classification")
    panel_class, confidence = predict_class(img)
    st.write(f"**Predicted Condition:** {panel_class} ({confidence:.2f} confidence)")

    st.subheader("2. Obstruction Detection with YOLOv8")
    temp_path = "D:\\Projects\\Guvi_Project5\\Intelligaurd\\temp_image.jpg"
    img.save(temp_path)
    det_result = predict_objects(temp_path)
    st.image(det_result.plot(), caption="Detected Objects", use_column_width=True)

    # Recommendations
    st.subheader("3. Maintenance Recommendation")
    if panel_class == 'Clean':
        st.success("No issues detected. Panel is in good condition.")
    else:
        st.warning(f"Detected issue: **{panel_class}**. Recommend cleaning or inspection.")

# === EDA & Metrics ===
st.subheader("🔍 EDA & Class Distributions")
if os.path.exists("D:\\Projects\\Guvi_Project5\\Intelligaurd\\data\\processed"):
    class_counts = {cls: len(os.listdir(f"data/processed/{cls}")) for cls in CLASS_NAMES if os.path.exists(f"data/processed/{cls}")}
    df_counts = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count'])
    st.bar_chart(df_counts)

st.subheader("📊 Model Performance Metrics")
if os.path.exists("D:\\Projects\\Guvi_Project5\\Intelligaurd\\runs\\detect\\solar_guard_detector3\\confusion_matrix.png"):
    st.image("D:\\Projects\\Guvi_Project5\\Intelligaurd\\runs\\detect\\solar_guard_detector3\\confusion_matrix.png", caption="YOLOv8 Confusion Matrix")

if os.path.exists("D:\\Projects\\Guvi_Project5\\Intelligaurd\\runs\detect\\solar_guard_detector3\\results.png"):
    st.image("D:\\Projects\\Guvi_Project5\\Intelligaurd\\runs\\detect\\solar_guard_detector3\\results.png", caption="Training Curves (YOLOv8)")

st.subheader("📈 Business Insights")
st.markdown("""
- **Frequent Issues**: Dust and Bird-Drop are the most common obstructions.
- **Actionable Insight**: Panels should be checked after storms or seasonal changes.
- **Efficiency Boost**: Early detection of obstructions prevents energy loss.
""")
