import torch
from tqdm import tqdm
from metrics import calculate_accuracy, calculate_auroc

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            total_samples += images.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            total_correct += calculate_accuracy(outputs, labels)
            
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
    epoch_loss = running_loss / total_samples
    epoch_acc = total_correct / total_samples
    epoch_auroc = calculate_auroc(torch.cat(all_outputs), torch.cat(all_labels))
    
    return epoch_loss, epoch_acc, epoch_auroc