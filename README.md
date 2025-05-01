```markdown
# ⚡ SolarGuard

**SolarGuard** is a deep learning-based system for real-time classification and detection of solar panel defects. It combines image classification (using CNN) and object detection (using YOLOv8) to monitor panel conditions, detect obstructions, and offer actionable maintenance recommendations.

---

## 🚀 Features

- 📷 **Image Upload** – Classify panel condition using CNN
- 🎯 **YOLOv8 Detection** – Detect and localize defects like dust, bird-drops, snow, etc.
- 📊 **EDA & Visualizations** – View class distributions, sample predictions, and training metrics
- 🧠 **Model Performance Metrics** – Accuracy, precision, recall, F1, mAP, confusion matrix
- 💡 **Business Insights** – Actionable recommendations for solar panel maintenance
- 🖥️ **Streamlit App** – Simple, interactive, and user-friendly interface

---

## 🧱 Tech Stack

- Python, Streamlit
- TensorFlow (MobileNetV2), YOLOv8 (Ultralytics)
- OpenCV, Matplotlib, Seaborn, Pandas

---

## 📁 Project Structure

```
SolarGuard/
├── app.py                      # Streamlit app
├── models/
│   └── mobilenetv2_panel_classifier.h5
├── runs/
│   └── detect/
│       └── solar_guard_detector2/
│           └── weights/
│               └── best.pt
├── data/
│   └── processed/              # For EDA
├── final_yolo_dataset/        # YOLO-formatted dataset
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone this Repository

```bash
git clone https://github.com/your-username/SolarGuard.git
cd SolarGuard
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🧪 Model Training

- Train YOLOv8 using:
```bash
python train_yolov8.py
```

- Evaluate:
```bash
python evaluate_metrics.py
python evaluate_and_visualize.py
```

---

## 📊 Model Performance

| Model         | Accuracy | mAP@0.5 | mAP@0.5:0.95 |
|---------------|----------|--------|--------------|
| MobileNetV2   | ~77%     | —      | —            |
| YOLOv8s       | —        | ✅     | ✅            |

---

## 💡 Business Insights

- Dust and bird-drops are the most common causes of performance degradation.
- Automated detection reduces manual inspection time.
- Preventive maintenance can be prioritized based on defect patterns.

---

## 📬 Contact

Created by [Your Name] – feel free to reach out!

```

---

Would you like me to add **badges**, GitHub deploy buttons, or customize it with your name and GitHub repo link?