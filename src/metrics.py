import torch

def calculate_accuracy(outputs, labels, threshold=0.5):
    """
    Calculates the batch accuracy for multi-label classification.
    This uses the "exact match ratio" where a prediction is correct only
    if all labels for that sample are correctly predicted.

    Args:
        outputs (torch.Tensor): The model's raw output logits.
        labels (torch.Tensor): The ground truth labels.
        threshold (float): The threshold for converting probabilities to binary predictions.

    Returns:
        int: The number of correctly predicted samples in the batch.
    """
    with torch.no_grad():
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()
        correct = (preds == labels).all(dim=1).sum().item()
        return correct