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
    current_size = 384 if model_name == "swin" else config.IMAGE_SIZE
    try:
        model = get_model()
        model.to(device)
        model.eval()
        
        # Simple input
        x = torch.randn(1, 3, current_size, current_size).to(device)
        
        # Current target layers from the plan
        target_name = None
        if model_name == "raddino":
            target_name = "encoder.encoder.layer.11.norm1"
        elif model_name == "radjepa":
            target_name = "encoder.model.blocks.11.norm1"
        elif model_name == "swin":
            target_name = "model.layers.3.blocks.1"
        elif model_name == "convnext":
            target_name = "model.stages.3.blocks.2"
        elif model_name == "cnn_transformer":
            target_name = "conv4"
        elif model_name == "efficientnet":
            target_name = "model.conv_head"
        elif model_name == "resnet50":
            target_name = "backbone.layer4"

        if not target_name:
            print(f"No target name for {model_name}")
            return

        modules = dict(model.named_modules())
        if target_name not in modules:
            print(f"Target {target_name} NOT FOUND. Available similar:")
            [print(n) for n in modules if target_name.split('.')[0] in n][:5]
            return

        target_layer = modules[target_name]
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
