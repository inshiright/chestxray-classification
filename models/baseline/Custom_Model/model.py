import torch
import torch.nn as nn

class CNNTransformerFromScratch(nn.Module):
    """
    Hybrid CNN + Transformer for multi-label chest X-ray classification.
    CNN backbone: 4 conv blocks with max pooling.
    Transformer: learns spatial dependencies between feature patches.
    Output: logits for 14 diseases.
    """
    def __init__(self, num_classes=14, img_size=224, in_channels=3, 
                 d_model=256, nhead=8, num_encoder_layers=4, 
                 dim_feedforward=512, dropout=0.1):
        super(CNNTransformerFromScratch, self).__init__()
        
        # ---------- CNN Backbone (from scratch) ----------
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )  # output: 56x56
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # output: 28x28
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # output: 14x14
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # output: 7x7
        
        self.feature_height = img_size // 32  # 224 / 32 = 7
        self.num_patches = self.feature_height * self.feature_height  # 7 * 7 = 49
        self.cnn_output_channels = 512
        
        # ---------- Projection to d_model ----------
        self.projection = nn.Linear(self.cnn_output_channels, d_model)
        
        # ---------- Positional encoding (learnable) ----------
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        # ---------- Transformer Encoder ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # ---------- Classification Head ----------
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)   # (B, 512, 7, 7)
        
        # Reshape to sequence
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C) = (B, 49, 512)
        
        # Project to d_model
        x = self.projection(x)            # (B, 49, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer(x)           # (B, 49, d_model)
        
        # Global average pooling over patches
        x = x.mean(dim=1)                 # (B, d_model)
        
        # Classification
        logits = self.classifier(x)       # (B, num_classes)
        return logits
