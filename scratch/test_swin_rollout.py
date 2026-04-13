import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import get_model
from src.xai_comparison import AttentionRecorder, load_trained_model
import src.config as config

print("Loading Swin...")
model = load_trained_model("swin")
tensor = torch.randn(1, 3, 384, 384).cuda()
print("Setting hooks...")
recorder = AttentionRecorder(model, "swin")
print("Forward pass...")
with torch.no_grad():
    model(tensor)
print(f"Captured {len(recorder.attentions)} attentions.")
