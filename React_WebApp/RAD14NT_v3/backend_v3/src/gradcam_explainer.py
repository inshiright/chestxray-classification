import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
from model import get_model
from config import IMAGE_SIZE, MODEL_NAME

# ---------------------------------------------------------
# Reshape functions for Transformer-based architectures
# ---------------------------------------------------------
def reshape_transform_vit(tensor):
    # Dynamically calculate grid size based on the number of patch tokens
    grid_size = int((tensor.size(1) - 1) ** 0.5)
    result = tensor[:, 1:, :].reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    
    # PyTorch Grad-CAM expects (Batch, Channels, Height, Width)
    result = result.permute(0, 3, 1, 2)
    return result

def reshape_transform_swin(tensor):
    # Swin norm1 hook outputs [B, H, W, C] — already spatial, just permute to [B, C, H, W]
    return tensor.permute(0, 3, 1, 2)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def run_gradcam(image_path, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    model.to(device)

    # Determine input size for this model
    if MODEL_NAME in ("raddino", "radjepa"):
        input_size = 518
    elif MODEL_NAME == "swin":
        input_size = 384
    else:
        input_size = IMAGE_SIZE  # 224

    # Dynamically assign the hook location based on the active model
    reshape_transform = None
    
    if MODEL_NAME == "efficientnet":
        # timm EfficientNet uses .model.blocks, not .model.features (torchvision convention)
        target_layers = [model.model.blocks[-1]]

    elif MODEL_NAME == "convnext":
        # timm ConvNeXt uses .model.stages, not .model.features
        target_layers = [model.model.stages[-1]]

    elif MODEL_NAME == "resnet50":
        # torchvision ResNet wrapped in ResNet50 class: access via model.backbone
        target_layers = [model.backbone.layer4[-1]]
        
    elif MODEL_NAME == "radjepa":
        target_layers = [model.encoder.model.blocks[-1].norm1]
        reshape_transform = reshape_transform_vit
        
    elif MODEL_NAME == "raddino":
        # Hugging Face native ViTModel structure for microsoft/rad-dino
        target_layers = [model.encoder.encoder.layer[-1].layernorm_before]
        reshape_transform = reshape_transform_vit

    elif MODEL_NAME == "swin":
        target_layers = [model.model.layers[-1].blocks[-1].norm1]
        reshape_transform = reshape_transform_swin
        
    else:
        raise ValueError(f"Grad-CAM configuration for {MODEL_NAME} is not set up.")

    cam = GradCAM(
        model=model, 
        target_layers=target_layers, 
        reshape_transform=reshape_transform
    )

    img = np.array(Image.open(image_path).convert('RGB'))
    img = cv2.resize(img, (input_size, input_size))
    rgb_img = np.float32(img) / 255

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Passing None to targets automatically explains the highest scoring class prediction
    targets = None
    
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(cam_image)
    plt.title(f'Grad-CAM Image Explainer ({MODEL_NAME})')
    plt.axis('off')
    plt.savefig(f'gradcam_output_{MODEL_NAME}.png')
    plt.show()

if __name__ == "__main__":
    # Example Usage:
    run_gradcam("../dataset/sample_xray.png", "path_to_saved_model.pth")