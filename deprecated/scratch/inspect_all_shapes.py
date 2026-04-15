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

def inspect_model(model_name):
    print(f"\n--- {model_name} ---")
    config.MODEL_NAME = model_name
    try:
        model = get_model()
        model.to(device)
        model.eval()
        
        # Simple input
        x = torch.randn(1, 3, 224, 224).to(device)
        
        # Find likely target layer
        target_layer = None
        target_name = None
        
        # Try some common patterns
        if model_name == "efficientnet":
            target_name = "features.8" if hasattr(model, 'features') else None
        elif model_name == "resnet50":
            target_name = "layer4"
        elif model_name == "convnext":
            target_name = "features.7"
        elif model_name == "swin":
            # Search for the last stage
            target_name = "features.6" 
        elif model_name == "raddino":
            target_name = "encoder.encoder.layer.11.norm1"
        elif model_name == "radjepa":
            target_name = "encoder.model.blocks.11.norm1"
        elif model_name == "cnn_transformer":
            target_name = "conv4"

        if target_name:
            modules = dict(model.named_modules())
            if target_name in modules:
                target_layer = modules[target_name]
            else:
                # If not exact, find closest
                for n in modules:
                    if target_name in n:
                        target_name = n
                        target_layer = modules[n]
                        break
        
        if not target_layer:
            print(f"Could not find target layer for {model_name}")
            return

        print(f"Targeting: {target_name} ({type(target_layer)})")
        
        activations = []
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations.append(output)
            
        handle = target_layer.register_forward_hook(hook)
        
        with torch.no_grad():
            model(x)
            
        handle.remove()
        
        if activations:
            act = activations[0]
            print(f"Shape: {act.shape}")
        else:
            print("No activations caught!")
            
    except Exception as e:
        print(f"Error inspecting {model_name}: {e}")

models = ["raddino", "radjepa", "swin", "efficientnet", "convnext", "cnn_transformer", "resnet50"]
for m in models:
    inspect_model(m)
