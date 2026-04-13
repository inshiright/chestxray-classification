import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from label_encoder import encode_labels
from torchvision import transforms
from config import IMAGE_SIZE

class NIHDataset(Dataset):
    def __init__(self, dataframe, is_train=False):
        self.df = dataframe.reset_index(drop=True)
        
        # Define base transforms (always applied)
        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]
        
        # Add augmentation ONLY if it is the training set
        if is_train:
            base_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
            
        # Final tensor conversion and normalization
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)

        labels = encode_labels(row["Finding Labels"])
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels