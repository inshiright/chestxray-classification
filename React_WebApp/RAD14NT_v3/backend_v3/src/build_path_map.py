import os
import pandas as pd
from tqdm import tqdm


def build_image_lookup(dataset_root):
    """
    Scan all images_XXX/images folders and build a lookup dict.
    """
    print("Scanning dataset for all image folders...")

    image_dict = {}

    # Loop through dataset folder
    for folder in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder)

        # Look for folders like images_001, images_002, etc.
        if folder.startswith("images_") and os.path.isdir(folder_path):
            images_subfolder = os.path.join(folder_path, "images")

            if os.path.exists(images_subfolder):
                print(f"Scanning: {images_subfolder}")

                for file in os.listdir(images_subfolder):
                    image_dict[file.lower()] = os.path.join(images_subfolder, file)

    print(f"Total images found: {len(image_dict)}")
    return image_dict


def build_dataframe_with_paths(csv_path, dataset_root):
    print("Loading CSV...")
    df = pd.read_csv(csv_path)

    if "Image Index" not in df.columns:
        raise ValueError(f"'Image Index' column not found. Columns: {df.columns}")

    # Build lookup from ALL folders
    image_lookup = build_image_lookup(dataset_root)

    print("Matching images...")

    paths = []
    for img_name in tqdm(df["Image Index"]):
        path = image_lookup.get(str(img_name).lower(), None)
        paths.append(path)

    df["image_path"] = paths

    # Remove rows where image not found
    df = df[df["image_path"].notnull()]

    print("Final dataset size:", len(df))

    return df