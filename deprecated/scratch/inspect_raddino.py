import torch
import sys
import os

# Add project root and src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if os.path.join(root_dir, 'src') not in sys.path:
    sys.path.append(os.path.join(root_dir, 'src'))

from model import get_model
import config

config.MODEL_NAME = "raddino"
model = get_model()

print("RadDINO Model Structure (last few layers):")
for name, module in model.named_modules():
    if "encoder.layer.11" in name or "norm" in name:
        print(name, "->", type(module))
