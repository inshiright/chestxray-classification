import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class RadJEPA(nn.Module):
    def __init__(self, num_classes=14, freeze_backbone=False):
        super(RadJEPA, self).__init__()
        
        # Load the self-supervised RadJEPA base encoder from Hugging Face
        self.encoder = AutoModel.from_pretrained(
            "AIDElab-IITBombay/RadJEPA", 
            trust_remote_code=True
        )
        
        # RadJEPA uses a ViT-B/14 backbone, which outputs a 768-dimensional feature vector
        hidden_dim = 768 
        
        # Linear classification head for our specific multi-label task
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Optional: Freeze the backbone for linear probing if needed later
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        # RadJEPA strictly expects 224x224. Interpolate if incoming batches are sized differently.
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        outputs = self.encoder(x)
        
        # Extract features (handling both HuggingFace BaseOutput and direct Tensor returns)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            features = outputs[0]
        else:
            features = outputs
            
        # RadJEPA processes the image into a sequence of tokens. 
        # We extract the CLS token (index 0) to represent the entire image for classification.
        if features.dim() == 3:
            cls_token = features[:, 0, :]
        else:
            cls_token = features
            
        logits = self.classifier(cls_token)
        return logits