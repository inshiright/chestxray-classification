import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_accuracy(outputs, labels, threshold=0.5):
    """
    Calculates the element-wise accuracy.
    """
    with torch.no_grad():
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()
        correct_per_sample = (preds == labels).float().mean(dim=1).sum().item()
        return correct_per_sample

def calculate_auroc(outputs, labels):
    """
    Calculates the macro AUROC score across all classes.
    """
    probs = torch.sigmoid(outputs).cpu().numpy()
    targets = labels.cpu().numpy()
    
    valid_auroc_scores = []
    for i in range(targets.shape[1]):
        # AUROC is only defined if there is at least one positive and one negative sample in the set
        if len(np.unique(targets[:, i])) > 1:
            score = roc_auc_score(targets[:, i], probs[:, i])
            valid_auroc_scores.append(score)
            
    return float(np.mean(valid_auroc_scores)) if valid_auroc_scores else 0.5