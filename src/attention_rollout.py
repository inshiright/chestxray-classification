import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import get_model
from config import IMAGE_SIZE, CHECKPOINT_DIR

def compute_rollout(attentions):
    # Create an identity matrix to represent the residual connections in the ViT
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    
    for attention in attentions:
        # Average the attention weights across all the attention heads in this layer
        attention_heads_fused = attention.mean(axis=1)
        
        # Add the identity matrix (residual connection)
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention.device)
        
        # Normalize the rows so they sum to 1
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        
        # Multiply this layer's attention with the accumulated attention from previous layers
        result = torch.matmul(attention_heads_fused, result)
        
    return result

def generate_attention_heatmap(image_path, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model and load your trained weights
    model = get_model()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Apply the exact same transformations used during validation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(original_image, (IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = transform(resized_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Specifically request the attention weights from the Hugging Face encoder
        outputs = model.encoder(input_tensor, output_attentions=True)
        attentions = outputs.attentions
        
        # We also want the actual predictions to see what the model diagnosed
        cls_token = outputs.pooler_output
        logits = model.classifier(cls_token)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Calculate the Attention Rollout across all layers
    rollout = compute_rollout(attentions)
    
    # The first token (index 0) is the CLS token. 
    # We slice [0, 0, 1:] to get the CLS token's attention to all the spatial image patches.
    cls_attention = rollout[0, 0, 1:]
    
    # RadDINO uses a patch size of 14. 
    # If IMAGE_SIZE is 518, the grid is 518/14 = 37. So we reshape into a 37x37 grid.
    grid_size = int(np.sqrt(cls_attention.size(0)))
    attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    
    # Normalize the heatmap between 0 and 1 for clean visualization
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
    
    # Resize the tiny 37x37 heatmap back up to the full 518x518 image size
    heatmap = cv2.resize(attention_map, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert it to a glowing colormap (Jet)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay the glowing heatmap onto the original X-ray
    overlay = cv2.addWeighted(resized_image, 0.5, heatmap_color, 0.5, 0)
    
    # --- Plotting ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original X-Ray")
    plt.imshow(resized_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Attention Heatmap")
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    save_path = "attention_rollout_result.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    print(f"Heatmap visualization saved to: {save_path}")
    print("\n--- Model Predictions ---")
    
    # Print the top 3 highest confidence diagnoses
    top_indices = np.argsort(probs)[::-1][:3]
    for idx in top_indices:
        print(f"Class {idx}: {probs[idx]:.4f} confidence")

if __name__ == "__main__":
    # Point this to any specific X-ray image you want to test
    image_to_test = "/content/dataset/images_01/images/00000001_000.png" 
    
    # Assumes your config.py has CHECKPOINT_DIR set correctly
    best_model_weights = os.path.join(CHECKPOINT_DIR, "raddino_best_model.pth")
    
    if os.path.exists(image_to_test) and os.path.exists(best_model_weights):
        generate_attention_heatmap(image_to_test, best_model_weights)
    else:
        print("Please update the paths to point to a valid image and your trained model.")