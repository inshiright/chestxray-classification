import sys
sys.path.insert(0, '.')
import torch
import config
from model import get_model

config.MODEL_NAME = "radjepa"
m = get_model()
print("RadJEPA:", m.encoder.__class__)

print("Looking for drop layers in radjepa:")
for n, module in m.encoder.named_modules():
    if "drop" in n and isinstance(module, torch.nn.Dropout):
        print("Drop layer:", n)

config.MODEL_NAME = "raddino"
m2 = get_model()
print("RadDINO:", m2.encoder.__class__)
print("Looking for drop layers in raddino:")
for n, module in m2.encoder.named_modules():
    if "drop" in n and ("attention" in n or "attn" in n) and isinstance(module, torch.nn.Dropout):
        print("Drop layer:", n)
        
