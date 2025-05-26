import os

# Paths
DATA_DIR = os.path.join("data", "train")
CHECKPOINT_DIR = os.path.join("outputs", "checkpoints")
PLOTS_DIR = os.path.join("outputs", "plots")

# Model settings
INPUT_SIZE = 144
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 30

# Learning Rate Scheduler
LR_INITIAL = 0.001
LR_DECAY_STEPS = 1000
LR_ALPHA = 0.1
