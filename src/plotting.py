import matplotlib.pyplot as plt
import os
from config import PLOTS_DIR


def plot_history(H):
    plt.figure(figsize=(12, 6))
    plt.plot(H.history["accuracy"], label="Train Accuracy", color="blue")
    plt.plot(H.history["val_accuracy"], label="Val Accuracy", color="green")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(H.history["loss"], label="Train Loss", color="red")
    plt.plot(H.history["val_loss"], label="Val Loss", color="orange")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(PLOTS_DIR, 'loss.png'))
    plt.show()
