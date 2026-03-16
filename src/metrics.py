import torch

def calculate_accuracy(outputs, labels, threshold=0.5):
    """
    Calculates the batch accuracy for multi-label classification.
    Instead of requiring a 14/14 exact match, this calculates the 
    element-wise accuracy (how many individual labels were predicted correctly).
    """
    with torch.no_grad():
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()
        
        # Calculate the average accuracy per sample (e.g. 13/14 correct = 0.92)
        # Then sum those up for the batch
        correct_per_sample = (preds == labels).float().mean(dim=1).sum().item()
        
        return correct_per_sample