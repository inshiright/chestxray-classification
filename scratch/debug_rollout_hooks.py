import torch
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path: sys.path.append(root_dir)
if os.path.join(root_dir, 'src') not in sys.path: sys.path.append(os.path.join(root_dir, 'src'))

from model import get_model
import config

def debug_hook(model_name):
    print(f"\n--- Debugging {model_name} ---")
    config.MODEL_NAME = model_name
    model = get_model()
    model.eval()
    
    captured = []
    def hook_fn(module, input, output):
        print(f"Hooked {module.__class__.__name__}")
        if isinstance(output, tuple):
            print(f"Output tuple elements: {[getattr(o, 'shape', type(o)) for o in output]}")
            if len(output) > 1 and torch.is_tensor(output[1]):
                captured.append(output[1])
        elif torch.is_tensor(output):
            print(f"Output tensor shape: {output.shape}")

    # Set hooks based on model_name
    if model_name == "radjepa":
        block = model.encoder.model.blocks[-1].attn
        handle = block.register_forward_hook(hook_fn)
    elif model_name == "cnn_transformer":
        attn = model.transformer.layers[-1].self_attn
        # Monkey patch forward
        orig_forward = attn.forward
        def new_forward(*args, **kwargs):
            kwargs['need_weights'] = True
            return orig_forward(*args, **kwargs)
        attn.forward = new_forward
        handle = attn.register_forward_hook(hook_fn)
    else:
        return

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        try:
            model(x, output_attentions=True)
        except:
            model(x)
    
    handle.remove()
    if captured:
        print(f"SUCCESS: Captured matrix with shape {captured[0].shape}")
    else:
        print("FAILURE: No matrix captured")

debug_hook("radjepa")
debug_hook("cnn_transformer")
