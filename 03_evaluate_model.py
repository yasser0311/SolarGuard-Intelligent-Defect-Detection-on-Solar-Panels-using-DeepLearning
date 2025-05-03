import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
MODEL_PATH = "D:\\Projects\\Guvi_Project5\\SolarGuard\\models\\mobilenetv2_panel_classifier.h5"
VAL_DIR = "D:\\Projects\\Guvi_Project5\\SolarGuard\\data\\final\\val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load validation data
val_gen = ImageDataGenerator(rescale=1./255)
val_data = val_gen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Get predictions
pred_probs = model.predict(val_data)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_data.classes
class_names = list(val_data.class_indices.keys())

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
