import os

# --- Paths ---
# On Kaggle, datasets live under /kaggle/input/. Locally, we'll point this
# to wherever you download a small sample later.
DATA_DIR = "/kaggle/input/datasets/khyeh0719/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
RESULTS_DIR = "results"

# --- The 5 diagnostic superclasses we're predicting ---
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = len(CLASSES)

# --- Training settings ---
SAMPLE_RATE = 100        # PTB-XL signals come in 100Hz or 500Hz versions; we use the lighter one
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_SEED = 42