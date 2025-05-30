import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from config import INPUT_SIZE, NUM_CLASSES, TRAIN_DIR

le = LabelEncoder()


def load_data():
    data, labels = [], []
    for i, path in enumerate(glob.glob(os.path.join(TRAIN_DIR, '*', '*'))):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = preprocess_input(img)
        data.append(img)

        label = os.path.basename(os.path.dirname(path))
        labels.append(label)

        # label = path.split("\\")[-2]
        # labels.append(label)

        if i % 2870 == 0:
            print(f"{i}/28708 samples loaded")

    data = np.array(data)
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, NUM_CLASSES)

    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    return x_train, x_val, y_train, y_val, le
