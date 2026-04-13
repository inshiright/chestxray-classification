import sys
sys.path.insert(0, '..')
import torch
import config
from src.model import get_model

config.MODEL_NAME = "cnn_transformer"
model = get_model().eval()

class AttentionWrapper(torch.nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.original_attn = original_attn
    def forward(self, query, key, value, **kwargs):
        kwargs['need_weights'] = True
        return self.original_attn(query, key, value, **kwargs)
    def __getattr__(self, name):
        if name in ['original_attn', '_modules', '_parameters', '_buffers']:
            return super().__getattr__(name)
        return getattr(self.original_attn, name)

for layer in model.transformer.layers:
    layer.self_attn = AttentionWrapper(layer.self_attn)

try:
    x = torch.randn(1, 3, 224, 224)
    model(x)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
