import torch
import sys
import os
import numpy as np

# Add project root and src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if os.path.join(root_dir, 'src') not in sys.path:
    sys.path.append(os.path.join(root_dir, 'src'))

from model import get_model
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inspect_model(model_name, target_layer_path):
    print(f"\n--- Inspecting {model_name} ---")
    config.MODEL_NAME = model_name
    model = get_model()
    model.to(device)
    model.eval()
    
    # Simple input
    x = torch.randn(1, 3, 224, 224).to(device)
    
    # Get the target layer
    target_layer = dict([*model.named_modules()])[target_layer_path]
    
    activations = []
    def hook(module, input, output):
        # HuggingFace layers sometimes return tuples
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output)
        
    handle = target_layer.register_forward_hook(hook)
    
    with torch.no_grad():
        model(x)
        
    handle.remove()
    
    if activations:
        act = activations[0]
        print(f"Activation shape: {act.shape}")
        num_tokens = act.size(1)
        hidden_dim = act.size(2)
        print(f"Num tokens: {num_tokens}, Hidden dim: {hidden_dim}")
        
inspect_model("raddino", "encoder.encoder.layer.11.norm1")
inspect_model("radjepa", "encoder.model.blocks.11.norm1")
inspect_model("swin", "features.7.2") # Last block of last stage?
