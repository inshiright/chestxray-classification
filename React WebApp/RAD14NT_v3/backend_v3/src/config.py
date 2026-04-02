import os

# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
CSV_PATH = os.path.join(ROOT_DIR, "dataset", "Data_Entry_2017.csv")
IMAGE_DIR = os.path.join(ROOT_DIR, "dataset")

# Model settings
MODEL_NAME = "efficientnet"
MODEL_TYPE = "generic_cv"
NUM_CLASSES = 14

# Training settings
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4
IMAGE_SIZE = 224

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
RESUME_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")  # Default path for resuming training