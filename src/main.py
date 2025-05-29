import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from config import Learning_Rate
from data_loader import load_data
from augment import get_augmentation
from model import build_model
from training import train_model
from evaluation import evaluate_model
from plotting import plot_history


def main():
    print("🔹 Loading data...")
    x_train, x_val, y_train, y_val, le = load_data()

    print("🔹 Applying data augmentation...")
    datagen = get_augmentation(x_train)

    print("🔹 Building model...")
    model = build_model(Learning_Rate)

    print("🔹 Starting training...")
    history = train_model(model, datagen, x_train, y_train, x_val, y_val)

    print("🔹 Evaluating model...")
    evaluate_model(model, x_val, y_val, le)

    print("🔹 Plotting history...")
    plot_history(history)


if __name__ == "__main__":
    main()
