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
import matplotlib.pyplot as plt

from build_path_map import build_dataframe_with_paths
from dataset_loader import NIHDataset
from data_split import split_data
from config import *
from train import train_one_epoch
from validate import validate_one_epoch
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = build_dataframe_with_paths()
train_df, val_df, test_df = split_data(df)

train_dataset = NIHDataset(train_df)
val_dataset = NIHDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, prefetch_factor=2)

model = get_model().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Early Stopping and Checkpoint Variables ---
start_epoch = 0
epoch_train_losses, epoch_val_losses = [], []
epoch_train_accuracies, epoch_val_accuracies = [], []
best_val_loss = float('inf')
patience_counter = 0
EARLY_STOPPING_PATIENCE = 5  # Stop after 5 epochs with no improvement

# Directory to save model checkpoints
checkpoint_dir = os.path.join(root_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, f"{MODEL_NAME}_best_model.pth")

# --- Resume Training from Checkpoint ---
# To resume, set this to the path of the 'latest_checkpoint.pth' file.
RESUME_CHECKPOINT_PATH = None 
if RESUME_CHECKPOINT_PATH and os.path.exists(RESUME_CHECKPOINT_PATH):
    print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT_PATH}")
    checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=device)
    
    # Check if the checkpoint is in the new comprehensive format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1 # Start from the next epoch
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore history
        epoch_train_losses = checkpoint.get('train_losses', [])
        epoch_val_losses = checkpoint.get('val_losses', [])
        epoch_train_accuracies = checkpoint.get('train_accuracies', [])
        epoch_val_accuracies = checkpoint.get('val_accuracies', [])
        
        print(f"Resumed from epoch {start_epoch}. Best validation loss: {best_val_loss:.4f}")
    else:
        print("Older checkpoint format detected (model weights only). Starting from epoch 0.")
        model.load_state_dict(checkpoint)

for epoch in range(start_epoch, EPOCHS):
    # --- Training ---
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    
    # --- Validation ---
    val_loss, val_acc = validate_one_epoch(
        model, val_loader, criterion, device
    )
    
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    epoch_train_losses.append(train_loss)
    epoch_val_losses.append(val_loss)
    epoch_train_accuracies.append(train_acc)
    epoch_val_accuracies.append(val_acc)

    # --- Checkpoint and Early Stopping Logic ---
    # Save the latest state of the training for resumption
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': epoch_train_losses,
        'val_losses': epoch_val_losses,
        'train_accuracies': epoch_train_accuracies,
        'val_accuracies': epoch_val_accuracies,
    }, latest_checkpoint_path)

    # Early stopping based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model weights separately
        torch.save(model.state_dict(), best_model_path)
        print(f"Validation loss improved. Saved best model to {best_model_path}")
    else:
        patience_counter += 1
        print(f"Validation loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered.")
        break

# --- Visualization Code ---
epochs_ran = len(epoch_train_losses)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(range(1, epochs_ran + 1), epoch_train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
ax1.plot(range(1, epochs_ran + 1), epoch_val_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
ax1.set_title(f'Loss per Epoch ({MODEL_NAME})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_xticks(range(1, epochs_ran + 1))
ax1.grid(True)
ax1.legend()

ax2.plot(range(1, epochs_ran + 1), epoch_train_accuracies, marker='o', linestyle='-', color='b', label='Training Accuracy')
ax2.plot(range(1, epochs_ran + 1), epoch_val_accuracies, marker='o', linestyle='-', color='r', label='Validation Accuracy')
ax2.set_title(f'Accuracy per Epoch ({MODEL_NAME})')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (Exact Match Ratio)')
ax2.set_xticks(range(1, epochs_ran + 1))
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# Save the plot as an image and display it
plt.savefig(f'training_curves_{MODEL_NAME}.png')
plt.show()

print("Training complete.")
if patience_counter >= EARLY_STOPPING_PATIENCE:
    print(f"Best model (epoch {epochs_ran - EARLY_STOPPING_PATIENCE}) saved at {best_model_path}")
else:
    print(f"Best model saved at {best_model_path}")