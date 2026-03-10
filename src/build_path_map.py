import os
import pandas as pd
from config import DATASET_DIR, CSV_PATH

def build_dataframe_with_paths():
    base_path = DATASET_DIR
    path_map = {}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".png"):
                path_map[file] = os.path.join(root, file)

    df = pd.read_csv(CSV_PATH)
    df['image_path'] = df['Image Index'].map(path_map)
    df['image_path'] = df['image_path'].astype(str).str.replace('\\','/',regex=False)
    missing = df['image_path'].isna().sum()
    print(f"Missing image paths: {missing}/{len(df)}")

    if missing > 0:
        missing_files = df.loc[df['image_path'].isna(),'Image Index'].unique()
        print("Missing filenames:", missing_files[:20])
        raise FileNotFoundError("Some images missing")

    return df