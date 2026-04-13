import timm
import torch.nn as nn

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=14, freeze_backbone=False):
        super().__init__()

        # Swin Transformer Base pretrained on ImageNet-22k, fine-tuned on ImageNet-1k at 384x384
        # Outputs 1024-dimensional feature vector after global average pooling
        self.model = timm.create_model(
            "swin_base_patch4_window12_384",
            pretrained=True,
            num_classes=num_classes
        )

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)
