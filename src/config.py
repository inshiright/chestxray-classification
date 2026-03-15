import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_DIR = "/content/dataset"
CSV_PATH = os.path.join(DATASET_DIR, "Data_Entry_2017.csv")

IMAGE_SIZE = 224
NUM_CLASSES = 14
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-4

MODEL_NAME = "radjepa"

# --- Checkpoint Settings ---
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# Change this to a string path to resume training (e.g., os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth"))
RESUME_CHECKPOINT_PATH = None