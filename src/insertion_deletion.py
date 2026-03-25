import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from torchvision import transforms
from model import get_model
from config import IMAGE_SIZE, CHECKPOINT_DIR

def compute_rollout(attentions):
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    for attention in attentions:
        attention_heads_fused = attention.mean(axis=1)
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention.device)
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        result = torch.matmul(attention_heads_fused, result)
    return result

def get_heatmap(model, input_tensor):
    outputs = model.encoder(input_tensor, output_attentions=True)
    attentions = outputs.attentions
    
    cls_token = outputs.pooler_output
    logits = model.classifier(cls_token)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    
    rollout = compute_rollout(attentions)
    cls_attention = rollout[0, 0, 1:]
    
    grid_size = int(np.sqrt(cls_attention.size(0)))
    attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
    
    heatmap = cv2.resize(attention_map, (IMAGE_SIZE, IMAGE_SIZE))
    return heatmap, probs

def calculate_insertion_deletion(image_path, model_weights_path, steps=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(original_image, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Baseline image tensor
    input_tensor = transform(resized_image).unsqueeze(0).to(device)
    
    # Get the heatmap and base predictions
    heatmap, base_probs = get_heatmap(model, input_tensor)
    
    # Identify the highest confidence pathology to track
    target_class_idx = np.argmax(base_probs)
    base_confidence = base_probs[target_class_idx]
    print(f"Tracking Class {target_class_idx} (Base Confidence: {base_confidence:.4f})")

    # Flatten the heatmap to sort pixels by importance
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1] # Highest to lowest importance
    total_pixels = len(sorted_indices)
    
    # Create a neutral baseline (mean pixel values) to act as the "blacked out" state
    mean_color = np.array([0.485, 0.456, 0.406]) * 255
    neutral_image = np.full_like(resized_image, mean_color, dtype=np.uint8)

    deletion_scores = []
    insertion_scores = []
    fractions = np.linspace(0, 1.0, steps + 1)

    with torch.no_grad():
        for frac in fractions:
            num_pixels_to_modify = int(frac * total_pixels)
            pixels_to_modify = sorted_indices[:num_pixels_to_modify]
            
            # --- Deletion: Start original, block out important pixels ---
            del_img = resized_image.copy().reshape(-1, 3)
            del_img[pixels_to_modify] = neutral_image.reshape(-1, 3)[0]
            del_img = del_img.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
            
            del_tensor = transform(del_img).unsqueeze(0).to(device)
            del_logits = model(del_tensor)
            del_prob = torch.sigmoid(del_logits)[0, target_class_idx].item()
            deletion_scores.append(del_prob)
            
            # --- Insertion: Start neutral, reveal important pixels ---
            ins_img = neutral_image.copy().reshape(-1, 3)
            ins_img[pixels_to_modify] = resized_image.reshape(-1, 3)[pixels_to_modify]
            ins_img = ins_img.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
            
            ins_tensor = transform(ins_img).unsqueeze(0).to(device)
            ins_logits = model(ins_tensor)
            ins_prob = torch.sigmoid(ins_logits)[0, target_class_idx].item()
            insertion_scores.append(ins_prob)

    # Calculate Area Under the Curve (AUC)
    # Good Deletion: Drops fast = Low AUC
    # Good Insertion: Rises fast = High AUC
    del_auc = auc(fractions, deletion_scores)
    ins_auc = auc(fractions, insertion_scores)

    print(f"Deletion AUC: {del_auc:.4f} (Lower is better)")
    print(f"Insertion AUC: {ins_auc:.4f} (Higher is better)")

    # --- Plot the Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(fractions * 100, deletion_scores, label=f'Deletion (AUC: {del_auc:.3f})', color='red', marker='o')
    plt.plot(fractions * 100, insertion_scores, label=f'Insertion (AUC: {ins_auc:.3f})', color='blue', marker='s')
    
    plt.axhline(y=base_confidence, color='gray', linestyle='--', alpha=0.5, label='Original Confidence')
    
    plt.title(f'Insertion/Deletion Test for Class {target_class_idx}')
    plt.xlabel('Percentage of Pixels Modified (%)')
    plt.ylabel('Model Confidence')
    plt.legend()
    plt.grid(True)
    
    save_path = "insertion_deletion_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    image_to_test = "/content/dataset/images_01/images/00000001_000.png" 
    best_model_weights = os.path.join(CHECKPOINT_DIR, "raddino_best_model.pth")
    
    if os.path.exists(image_to_test) and os.path.exists(best_model_weights):
        calculate_insertion_deletion(image_to_test, best_model_weights)
    else:
        print("Please update the paths to point to a valid image and your trained model.")