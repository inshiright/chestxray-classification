import torch
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path: sys.path.append(root_dir)
if os.path.join(root_dir, 'src') not in sys.path: sys.path.append(os.path.join(root_dir, 'src'))

from model import get_model
import config

def inspect_attentions(model_name):
    print(f"\n--- {model_name} ---")
    config.MODEL_NAME = model_name
    try:
        model = get_model()
        for name, module in model.named_modules():
            # Look for MultiheadAttention or self-attention blocks
            if "attn" in name.lower() and ("Block" in str(type(module)) or "Attention" in str(type(module))):
                print(f"Found potential: {name} ({type(module)})")
                if "blocks.11" in name or "layer.11" in name or "layers.3" in name:
                     print(f"MATCH: {name}")
    except Exception as e:
        print(f"Error: {e}")

models = ["raddino", "radjepa", "swin", "cnn_transformer"]
for m in models:
    inspect_attentions(m)
