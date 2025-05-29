import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model as model
from sklearn.utils import class_weight
from config import CHECKPOINT_DIR
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from callbacks.lr_scheduler import LogCosineDecay
from config import Learning_Rate, LR_log, EPOCHS, BATCH_SIZE


def train_model(model, datagen, x_train, y_train, x_val, y_val):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        LogCosineDecay(Learning_Rate, LR_log)
    ]

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    weights = dict(enumerate(weights))

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=weights
    )

    model.save(os.path.join(CHECKPOINT_DIR, "final_model.keras"))
    return history
