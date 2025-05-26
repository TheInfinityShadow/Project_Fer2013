from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from .config import CHECKPOINT_DIR


def train_model(model, datagen, x_train, y_train, x_val, y_val, callbacks):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=list(range(y_train.shape[1])),
        y=y_train.argmax(axis=1)
    )
    weights = dict(enumerate(weights))

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        validation_data=(x_val, y_val),
        epochs=30,
        callbacks=callbacks,
        class_weight=weights
    )

    model.save(os.path.join(CHECKPOINT_DIR, "final_model.h5"))
    return history
