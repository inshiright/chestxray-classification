import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_gradcam import GradCAM
from pytorch_gradcam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
from model import get_model
from config import IMAGE_SIZE, MODEL_NAME

# ---------------------------------------------------------
# Reshape functions for Transformer-based architectures
# ---------------------------------------------------------
def reshape_transform_vit(tensor, height=24, width=24):
    # RadJEPA/ViT models usually output a class token at index 0 followed by patch tokens.
    # For a 384x384 image with 16x16 patches, the grid is 24x24 (384 / 16 = 24).
    # We drop the class token [:, 1:, :] and reshape the remaining patches into a 2D grid.
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    
    # PyTorch Grad-CAM expects (Batch, Channels, Height, Width)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_swin(tensor, height=12, width=12):
    # Swin Transformers output flattened features without a class token.
    # For a 384x384 image, the final stage feature map is typically 12x12 (384 / 32 = 12).
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def run_gradcam(image_path, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    model.to(device)

    # Dynamically assign the hook location based on the active model
    reshape_transform = None
    
    if MODEL_NAME == "efficientnet":
        target_layers = [model.features[-1]]
        
    elif MODEL_NAME == "convnext":
        target_layers = [model.features[-1][-1]]
        
    elif MODEL_NAME == "radjepa":
        target_layers = [model.blocks[-1].norm1]
        reshape_transform = reshape_transform_vit
        
    elif MODEL_NAME == "swin":
        target_layers = [model.features[-1][-1]] 
        reshape_transform = reshape_transform_swin
        
    else:
        raise ValueError(f"Grad-CAM configuration for {MODEL_NAME} is not set up.")

    cam = GradCAM(
        model=model, 
        target_layers=target_layers, 
        reshape_transform=reshape_transform
    )

    img = np.array(Image.open(image_path).convert('RGB'))
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.float32(img) / 255

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Passing None to targets automatically explains the highest scoring class prediction
    targets = None
    
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