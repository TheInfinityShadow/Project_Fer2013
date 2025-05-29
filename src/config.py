import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.optimizers.schedules import CosineDecay


# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Data Paths
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")

# Output Paths
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "outputs", "label_encoder.pkl")

# Model
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "Emotion_Detector_Fer2013_V8.keras")

# Model settings
INPUT_SIZE = 144
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 30

# Learning Rate Scheduler
LR_INITIAL = 0.001
LR_DECAY_STEPS = 1000
LR_ALPHA = 0.1
LR_log = []

Learning_Rate = CosineDecay(initial_learning_rate=LR_INITIAL, decay_steps=LR_DECAY_STEPS, alpha=LR_ALPHA)


# Class Labels (index-based)
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
