import torch
import torch.nn as nn
from transformers import AutoModel

class RadDINO(nn.Module):
    def __init__(self, num_classes=14, freeze_backbone=True):
        super(RadDINO, self).__init__()
        
        # Load Microsoft's Rad-DINO chest X-ray foundation model
        self.encoder = AutoModel.from_pretrained("microsoft/rad-dino")
        
        # Rad-DINO uses a ViT-Base backbone, which outputs a 768-dimensional feature vector
        hidden_dim = 768 
        
        # Linear classification head mapping to your 14 target classes
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Freeze the backbone to only train the classification head
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
            # Ensure the new classification head explicitly requires gradients
            for param in self.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        outputs = self.encoder(x)
        
        # Rad-DINO natively provides the 'pooler_output' which acts as the global CLS token
        cls_token = outputs.pooler_output
            
        logits = self.classifier(cls_token)
        return logits