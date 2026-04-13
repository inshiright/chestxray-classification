import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import get_model
import src.config as config

print("Loading RadJEPA...")
config.MODEL_NAME = "radjepa"
try:
    model = get_model()
    print("Inspecting RadJEPA encoder...")
    for name, module in model.encoder.named_modules():
        if ("block" in name or "attention" in name or "drop" in name or "attn" in name) and isinstance(module, torch.nn.Module):
            print(name, type(module))
except Exception as e:
    import traceback
    traceback.print_exc()

print("Loading RadDINO...")
config.MODEL_NAME = "raddino"
try:
    model2 = get_model()
    print("Inspecting RadDINO encoder...")
    for name, module in model2.encoder.named_modules():
        if "layer.11" in name and ("attention" in name or "drop" in name or "attn" in name) and isinstance(module, torch.nn.Module):
            print(name, type(module))
except Exception as e:
    pass
