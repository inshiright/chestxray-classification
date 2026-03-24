import sys
import os

# Adds 'chestxray-classification' (repo root) and 'chestxray-classification/src' to the system path
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

train_dataset = NIHDataset(train_df, is_train=True)
val_dataset = NIHDataset(val_df, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, prefetch_factor=2)

model = get_model().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Scheduler tracks Validation AUROC now (mode='max' because higher is better)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# --- Early Stopping and Checkpoint Variables ---
start_epoch = 0
epoch_train_losses, epoch_val_losses = [], []
epoch_train_accuracies, epoch_val_accuracies = [], []
epoch_train_aurocs, epoch_val_aurocs = [], []

best_val_auroc = 0.0
patience_counter = 0
EARLY_STOPPING_PATIENCE = 5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best_model.pth")

if RESUME_CHECKPOINT_PATH and os.path.exists(RESUME_CHECKPOINT_PATH):
    print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT_PATH}")
    checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint.get('epoch', 0) + 1 
        best_val_auroc = checkpoint.get('best_val_auroc', 0.0)
        
        epoch_train_losses = checkpoint.get('train_losses', [])
        epoch_val_losses = checkpoint.get('val_losses', [])
        epoch_train_accuracies = checkpoint.get('train_accuracies', [])
        epoch_val_accuracies = checkpoint.get('val_accuracies', [])
        epoch_train_aurocs = checkpoint.get('train_aurocs', [])
        epoch_val_aurocs = checkpoint.get('val_aurocs', [])
        
        print(f"Resumed from epoch {start_epoch}. Best validation AUROC: {best_val_auroc:.4f}")
    else:
        print("Older checkpoint format detected. Starting from epoch 0.")
        model.load_state_dict(checkpoint)

for epoch in range(start_epoch, EPOCHS):
    train_loss, train_acc, train_auroc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    
    val_loss, val_acc, val_auroc = validate_one_epoch(
        model, val_loader, criterion, device
    )
    
    scheduler.step(val_auroc)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(
        f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f}\n"
        f"Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUROC: {train_auroc:.4f}\n"
        f"Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUROC: {val_auroc:.4f}\n"
        f"-"*60
    )

    epoch_train_losses.append(train_loss)
    epoch_val_losses.append(val_loss)
    epoch_train_accuracies.append(train_acc)
    epoch_val_accuracies.append(val_acc)
    epoch_train_aurocs.append(train_auroc)
    epoch_val_aurocs.append(val_auroc)

    latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_auroc': best_val_auroc,
        'train_losses': epoch_train_losses,
        'val_losses': epoch_val_losses,
        'train_accuracies': epoch_train_accuracies,
        'val_accuracies': epoch_val_accuracies,
        'train_aurocs': epoch_train_aurocs,
        'val_aurocs': epoch_val_aurocs,
    }, latest_checkpoint_path)

    # Early stopping evaluates based on AUROC now
    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Validation AUROC improved. Saved best model to {best_model_path}\n")
    else:
        patience_counter += 1
        print(f"Validation AUROC did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}\n")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered.")
        break

# --- Visualization Code ---
epochs_ran = len(epoch_train_losses)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

ax1.plot(range(1, epochs_ran + 1), epoch_train_losses, marker='o', linestyle='-', color='b', label='Train')
ax1.plot(range(1, epochs_ran + 1), epoch_val_losses, marker='o', linestyle='-', color='r', label='Val')
ax1.set_title(f'Loss ({MODEL_NAME})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.legend()

ax2.plot(range(1, epochs_ran + 1), epoch_train_accuracies, marker='o', linestyle='-', color='b', label='Train')
ax2.plot(range(1, epochs_ran + 1), epoch_val_accuracies, marker='o', linestyle='-', color='r', label='Val')
ax2.set_title(f'Accuracy ({MODEL_NAME})')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid(True)
ax2.legend()

ax3.plot(range(1, epochs_ran + 1), epoch_train_aurocs, marker='o', linestyle='-', color='b', label='Train')
ax3.plot(range(1, epochs_ran + 1), epoch_val_aurocs, marker='o', linestyle='-', color='r', label='Val')
ax3.set_title(f'AUROC ({MODEL_NAME})')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('AUROC Score')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.savefig(f'training_curves_{MODEL_NAME}.png')
plt.show()

print("Training complete.")
if patience_counter >= EARLY_STOPPING_PATIENCE:
    print(f"Best model (epoch {epochs_ran - EARLY_STOPPING_PATIENCE}) saved at {best_model_path}")
else:
    print(f"Best model saved at {best_model_path}")