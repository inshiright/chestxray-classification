import torch
import sys
import os
from transformers import AutoModel

# Load DinoV2 (RadDINO is based on this)
model_id = "facebook/dinov2-base"
model = AutoModel.from_pretrained(model_id)

attn = model.encoder.layer[0].attention
print(f"Attention module type: {type(attn)}")
print(f"Members: {dir(attn)}")

for name, module in attn.named_modules():
    print(f"Submodule: {name} ({type(module)})")
