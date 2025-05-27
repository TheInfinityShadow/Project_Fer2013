from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50V2
from cbam import CBAMLayer
from config import INPUT_SIZE, NUM_CLASSES
from loss import focal_loss


def build_model(learning_rate_schedule):
    base_model = ResNet50V2(include_top=False, weights="imagenet", input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

    for layer in base_model.layers[:-50]:
        layer.trainable = False

    inputs = layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = base_model(inputs, training=False)
    x = CBAMLayer()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2048, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.RMSprop(learning_rate=learning_rate_schedule),
                  loss=focal_loss(), metrics=["accuracy"])
    return model
