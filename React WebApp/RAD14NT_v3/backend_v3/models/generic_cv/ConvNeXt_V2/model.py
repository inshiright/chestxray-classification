import timm
import torch.nn as nn

class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=14, freeze_backbone=False):
        super().__init__()

        # ConvNeXt V2 Base pretrained on ImageNet-22k, fine-tuned on ImageNet-1k
        # Outputs 1024-dimensional feature vector after global average pooling
        self.model = timm.create_model(
            "convnextv2_base",
            pretrained=True,
            num_classes=num_classes
        )

        # Optional: Freeze the backbone for linear probing
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)
