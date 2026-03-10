import timm
import torch.nn as nn

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # `timm` may not have pretrained weights for all architectures:
        # - In some versions the weights for "efficientnetv2_s" are not available.
        # - Use pretrained=False to avoid runtime errors if you don't need pretrained weights.
        self.model = timm.create_model(
            "efficientnetv2_s",
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)