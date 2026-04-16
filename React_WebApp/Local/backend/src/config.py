import os

# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
CSV_PATH = os.path.join(ROOT_DIR, "dataset", "Data_Entry_2017.csv")
IMAGE_DIR = os.path.join(ROOT_DIR, "dataset")

# Model settings — override via environment variables for different training runs
MODEL_NAME = os.environ.get("MODEL_NAME", "efficientnet")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "generic_cv")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", 14))

# Training settings — all overridable via env vars
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
EPOCHS     = int(os.environ.get("EPOCHS", 5))
LR         = float(os.environ.get("LR", 1e-4))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 224))

# Checkpoint directory
CHECKPOINT_DIR = os.environ.get(
    "CHECKPOINT_DIR",
    os.path.join(ROOT_DIR, "checkpoints"),
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
RESUME_CHECKPOINT_PATH = os.environ.get(
    "RESUME_CHECKPOINT_PATH",
    os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth"),
)