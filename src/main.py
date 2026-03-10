import sys
import os

# Adds 'chestxray-classification' (repo root) and 'chestxray-classification/src' to the system path
# so local imports like `from models...` work when running this file directly.
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(root_path, 'src')
for p in (root_path, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import pandas as pd
from torch.utils.data import DataLoader

from build_path_map import build_dataframe_with_paths
from dataset_loader import NIHDataset
from data_split import split_data
from config import *
from train import train_one_epoch
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = build_dataframe_with_paths()
train_df, val_df, test_df = split_data(df)

train_dataset = NIHDataset(train_df)
val_dataset = NIHDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

model = get_model().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )

    print(f"Epoch {epoch+1} Train Loss: {train_loss}")