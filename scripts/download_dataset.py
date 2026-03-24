import os
import subprocess
import zipfile
from tqdm import tqdm

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
    members = zip_ref.namelist()
    for member in tqdm(members, desc="Extracting", unit="file"):
        zip_ref.extract(member, DATA_DIR)

# Uncomment the following line if you want to remove the zip file after extraction
# os.remove(zip_path)

print("Dataset ready.")