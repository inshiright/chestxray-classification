import torch
import sys
sys.path.insert(0, '..')
import config
from src.model import get_model

config.MODEL_NAME = "radjepa"
m = get_model().eval()

captured = []
def hook(module, input, output):
    print("Hook triggered on", module)
    captured.append(output)

print("Registering hooks")
for block in m.encoder.model.blocks:
    block.attn.attn_drop.register_forward_hook(hook)
    
x = torch.randn(1, 3, 224, 224)
print("Forward pass...")
with torch.no_grad():
    m(x)

print(f"Captured {len(captured)} tensors")
if len(captured) > 0:
    print(captured[0].shape)
