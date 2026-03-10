import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
CSV_PATH = os.path.join(DATASET_DIR, "Data_Entry_2017.csv")

IMAGE_SIZE = 384
NUM_CLASSES = 14
BATCH_SIZE = 8  # if OOM on laptop GPU, lower this value (e.g. 8 or 4).
EPOCHS = 10
LR = 1e-4

MODEL_NAME = "efficientnet"