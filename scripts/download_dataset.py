import os
import subprocess
import zipfile

DATASET_NAME = "nih-chest-xrays/data"
DATA_DIR = "dataset"

print("Downloading NIH Chest X-ray dataset...")

subprocess.run(
    ["kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", DATA_DIR],
    check=True,
)

print("Extracting dataset...")

zip_path = os.path.join(DATA_DIR, "data.zip")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

os.remove(zip_path)

print("Dataset ready.")