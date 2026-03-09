import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

print("Is CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")