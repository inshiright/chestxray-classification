import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ResNet50, self).__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
