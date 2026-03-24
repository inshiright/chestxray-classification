import torch
from tqdm import tqdm
from metrics import calculate_accuracy, calculate_auroc

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_outputs = []
    all_labels = []

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        total_samples += images.size(0)

        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_correct += calculate_accuracy(outputs, labels)
        
        # Detach and move to CPU immediately to prevent GPU memory leaks
        all_outputs.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    epoch_loss = running_loss / total_samples
    epoch_acc = total_correct / total_samples
    
    # Calculate global AUROC for the epoch
    epoch_auroc = calculate_auroc(torch.cat(all_outputs), torch.cat(all_labels))
    
    return epoch_loss, epoch_acc, epoch_auroc