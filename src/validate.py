import torch
from tqdm import tqdm
from metrics import calculate_accuracy

def validate_one_epoch(model, val_loader, criterion, device):
    """
    Validates the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): The validation data loader.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to run the validation on.
        
    Returns:
        tuple: A tuple containing:
            - float: The average validation loss for the epoch.
            - float: The average validation accuracy for the epoch.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            total_samples += images.size(0)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total_correct += calculate_accuracy(outputs, labels)
    epoch_loss = running_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc