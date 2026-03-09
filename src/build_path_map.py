import os
import pandas as pd

# Path to your 'dataset' folder from the image
base_path = "../dataset" 

# Create a map: { "filename.png": "full/path/to/filename.png" }
path_map = {}
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".png"):
            path_map[file] = os.path.join(root, file)

# Now, when you load your CSV, you can add a 'full_path' column
df = pd.read_csv("../dataset/Data_Entry_2017.csv")
df['image_path'] = df['Image Index'].map(path_map)

# Normalise paths to only use forward slashes
df['image_path'] = df['image_path'].astype(str).str.replace('\\', '/', regex=False)

# Check for missing paths (i.e. if any filenames were not found)
missing = df['image_path'].isna().sum()
print(f"Missing image paths: {missing} / {len(df)}")

if missing > 0:
    missing_files = df.loc[df['image_path'].isna(), 'Image Index'].unique()
    print("Missing filenames (first 20):", missing_files[:20].tolist())
    raise FileNotFoundError(f"{missing} image(s) from the CSV were not found under {base_path}.")

# Example: Full path for the first 5 images
# print(df['image_path'].head())