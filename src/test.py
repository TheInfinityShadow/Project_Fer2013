import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .config import TEST_DIR, MODEL_PATH, LABELS, INPUT_SIZE
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from .cbam import CBAMLayer
import seaborn as sns
import numpy as np
import glob
import cv2


# ------------------ Load Model ------------------ #
model = load_model(MODEL_PATH, compile=False, custom_objects={"CBAMLayer": CBAMLayer})

# ------------------ Load Test Images ------------------ #
X_test, y_true = [], []

for i, label in enumerate(LABELS):
    path = os.path.join(TEST_DIR, label, "*")
    files = glob.glob(path)

    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = preprocess_input(img)
        X_test.append(img)
        y_true.append(i)

X_test = np.array(X_test)
y_true = np.array(y_true)

# ------------------ Prediction ------------------ #

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# ------------------ Evaluation ------------------ #
cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))

# Accuracy per class
print("\n🎯 Accuracy per class:")
for i, label in enumerate(LABELS):
    total = cm[i].sum()
    correct = cm[i][i]
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"{label:>10}: {acc:.2f}%  ({correct}/{total})")

# Confusion Matrix
print("\n📊 Confusion Matrix:")
print(cm)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\n📝 Classification Report:")
print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))
