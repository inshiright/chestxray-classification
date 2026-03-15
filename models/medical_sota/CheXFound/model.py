import torch
import torch.nn as nn
from transformers import AutoModel

class CheXFound(nn.Module):
    def __init__(self, num_classes=14, freeze_backbone=True):
        super(CheXFound, self).__init__()
        
        # Load the CheXFound base encoder from Hugging Face
        self.encoder = AutoModel.from_pretrained(
            "DIAL-RPI/CheXFound", 
            trust_remote_code=True
        )
        
        # CheXFound uses a ViT-Large backbone, which outputs a 1024-dimensional feature vector
        hidden_dim = 1024 
        
        # Linear classification head mapping to your 14 target classes
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Freeze the backbone for linear probing / fine-tuning only the head
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
            # Ensure the new classification head explicitly requires gradients
            for param in self.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        outputs = self.encoder(x)
        
        # Extract features depending on the Hugging Face return type
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            features = outputs[0]
        else:
            features = outputs
            
        # Extract the CLS token (index 0) to represent the global image
        if features.dim() == 3:
            cls_token = features[:, 0, :]
        else:
            cls_token = features
            
        logits = self.classifier(cls_token)
        return logits