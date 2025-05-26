from src.data_loader import load_data
from src.augment import get_augmentation
from src.model import build_model
from src.training import train_model
from src.evaluation import evaluate_model
from src.plotting import plot_history
from callbacks.lr_scheduler import LogCosineDecay
from tensorflow.keras.optimizers.schedules import CosineDecay
from src.config import LR_INITIAL, LR_DECAY_STEPS, LR_ALPHA

lr_log = []
lr_schedule = CosineDecay(initial_learning_rate=LR_INITIAL, decay_steps=LR_DECAY_STEPS, alpha=LR_ALPHA)

(x_train, x_val, y_train, y_val), le = load_data()
datagen = get_augmentation(x_train)
model = build_model(lr_schedule)

callbacks = [LogCosineDecay(lr_schedule, lr_log)]
history = train_model(model, datagen, x_train, y_train, x_val, y_val, callbacks)
evaluate_model(model, x_val, y_val, le)
plot_history(history)
