import matplotlib.pyplot as plt
import os
from config import PLOTS_DIR


def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(PLOTS_DIR, 'loss.png'))
    plt.show()
