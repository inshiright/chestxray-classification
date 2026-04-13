import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import functools

# Add project root and src directory to path to access models and config
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)

import config
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration ---
# Note: Paths are relative to the 'src' directory
MODEL_WEIGHTS = {
    "raddino": "../model_checkpoints/raddino_best_model.pth",
    "radjepa": "../model_checkpoints/radjepa_best_model.pth",
    "swin": "../model_checkpoints/swin_best_model.pth",
    "efficientnet": "../model_checkpoints/efficientnet_best_model.pth",
    "convnext": "../model_checkpoints/convnext_best_model.pth",
    "cnn_transformer": "../model_checkpoints/cnn_transformer_best_model.pth",
    "resnet50": "../model_checkpoints/resnet50_best_model.pth"
}

DEFAULT_TEST_IMAGE = "../dataset/00000001_002.png"
IMG_DIR = r"C:\Users\Rald999\Documents\GitHub\chestxray-classification\img"
os.makedirs(IMG_DIR, exist_ok=True)

# Utility functions for Grad-CAM with Transformers
def reshape_transform_vit(tensor):
    n_tokens = tensor.size(1)
    if int(np.sqrt(n_tokens - 1))**2 == n_tokens - 1:
        grid_size = int(np.sqrt(n_tokens - 1))
        result = tensor[:, 1:, :].reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    else:
        grid_size = int(np.sqrt(n_tokens))
        result = tensor.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    
    result = result.permute(0, 3, 1, 2)
    return result

def reshape_transform_h_w_c(tensor):
    return tensor.permute(0, 3, 1, 2)

def load_trained_model(model_name):
    # Temporarily override MODEL_NAME in config to use get_model()
    original_name = config.MODEL_NAME
    config.MODEL_NAME = model_name
    
    try:
        model = get_model()
        weights_path = os.path.join(script_dir, MODEL_WEIGHTS.get(model_name, ""))
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded {model_name} weights from {weights_path}")
        else:
            print(f"Warning: No weights found for {model_name} at {weights_path}. Using random initialization.")
        
        # Unfreeze all parameters for XAI analysis (Grad-CAM requires gradients)
        for param in model.parameters():
            param.requires_grad = True
            
        model.to(device)
        model.eval()
        return model
    finally:
        config.MODEL_NAME = original_name

def get_preprocess_transform(size=config.IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_image_tensor(image_path, size=config.IMAGE_SIZE):
    img = Image.open(image_path).convert('RGB')
    transform = get_preprocess_transform(size)
    return transform(img).unsqueeze(0).to(device)

def generate_all_gradcam(image_path, model_names):
    print("Generating Grad-CAM comparative view...")
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))
    
    img = Image.open(image_path).convert('RGB')
    axes[0].imshow(img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    axes[0].set_title("Original X-Ray")
    axes[0].axis('off')
    
    # Default display size for original image
    axes[0].imshow(img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))

    for i, model_name in enumerate(tqdm(model_names, desc="Generating Grad-CAM")):
        # Handle models with different input size requirements
        current_size = 384 if model_name == "swin" else config.IMAGE_SIZE
        model = load_trained_model(model_name)
        input_tensor = get_image_tensor(image_path, size=current_size)
        rgb_img_model = np.float32(img.resize((current_size, current_size))) / 255
        
        reshape_transform = None
        if model_name == "efficientnet":
            target_layers = [model.model.conv_head]
        elif model_name == "convnext":
            target_layers = [model.model.stages[-1].blocks[-1]]
        elif model_name == "resnet50":
            target_layers = [model.backbone.layer4[-1]] 
        elif model_name == "raddino":
            target_layers = [model.encoder.encoder.layer[-1].norm1]
            reshape_transform = reshape_transform_vit
        elif model_name == "radjepa":
            target_layers = [model.encoder.model.blocks[-1].norm1]
            reshape_transform = reshape_transform_vit
        elif model_name == "swin":
            target_layers = [model.model.layers[-1].blocks[-1]]
            reshape_transform = reshape_transform_h_w_c
        elif model_name == "cnn_transformer":
            target_layers = [model.conv4[-1]]
        else:
            print(f"Skipping {model_name}: Grad-CAM layers not configured.")
            continue

        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img_model, grayscale_cam, use_rgb=True)
        
        # Save individual image
        ind_fig, ind_ax = plt.subplots(figsize=(6, 6))
        ind_ax.imshow(cam_image)
        ind_ax.set_title(f"Grad-CAM: {model_name}")
        ind_ax.axis('off')
        ind_fig.savefig(os.path.join(IMG_DIR, f"{model_name}_gradcam.png"), bbox_inches='tight')
        plt.close(ind_fig)

        axes[i+1].imshow(cam_image)
        axes[i+1].set_title(f"Grad-CAM: {model_name}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "gradcam_comparison.png"))
    plt.show()

def compute_rollout(attentions):
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    for attention in attentions:
        attention_heads_fused = attention.mean(axis=1)
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention.device)
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        result = torch.matmul(attention_heads_fused, result)
    return result

class AttentionRecorder:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.attentions = []
        self.hooks = []
        self._set_hooks()

    def _hook(self, module, input, output):
        # Handle different output formats of attention layers
        with open("hook_debug.log", "a") as f:
            if isinstance(output, tuple):
                # Many implementations (HF, some timm, torch.nn.MHA) return (output, attn_weights)
                if len(output) > 1 and output[1] is not None:
                    attn = output[1].detach()
                    if attn.dim() == 3: # (B, N, N) -> (B, 1, N, N)
                        attn = attn.unsqueeze(1)
                    self.attentions.append(attn)
                    f.write(f"HOOK: tuple on {module.__class__.__name__}, shape: {attn.shape}\n")
            elif isinstance(output, torch.Tensor):
                # If the hook is on a module that only outputs the matrix, like attn_drop
                if output.dim() == 4: # (B, heads, N, N)
                    self.attentions.append(output.detach())
                    f.write(f"HOOK: tensor on {module.__class__.__name__}, shape: {output.shape}\n")
                elif output.dim() == 3: # (B, N, N)
                    self.attentions.append(output.unsqueeze(1).detach())
                    f.write(f"HOOK: 3D tensor on {module.__class__.__name__}, shape: {output.shape}\n")

    def _set_hooks(self):
        # Target internal attention modules based on architecture
        if self.model_name == "raddino":
            # HF DinoV2: dropout layer runs on attention weights matrix
            for block in self.model.encoder.encoder.layer:
                if hasattr(block.attention.attention, 'dropout'):
                    self.hooks.append(block.attention.attention.dropout.register_forward_hook(self._hook))
                else:
                    self.hooks.append(block.attention.attention.register_forward_hook(self._hook))
        elif self.model_name == "radjepa":
            # Timm ViT: attn_drop layer
            for block in self.model.encoder.model.blocks:
                # Disable fused_attn so that attn_drop is explicitly called in the forward pass
                if hasattr(block.attn, 'fused_attn'):
                    block.attn.fused_attn = False
                if hasattr(block.attn, 'attn_drop'):
                    self.hooks.append(block.attn.attn_drop.register_forward_hook(self._hook))
                else:
                    self.hooks.append(block.attn.register_forward_hook(self._hook))
        elif self.model_name == "swin":
            # Swin: attn_drop within window attention
            for layer in self.model.model.layers:
                for block in layer.blocks:
                    if hasattr(block.attn, 'attn_drop'):
                        self.hooks.append(block.attn.attn_drop.register_forward_hook(self._hook))
                    else:
                        self.hooks.append(block.attn.register_forward_hook(self._hook))
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def generate_all_rollout(image_path, model_names):
    print("Generating Attention Rollout comparative view...")
    # Filter for models that have attention mechanisms
    transformer_list = ["raddino", "radjepa", "swin", "cnn_transformer"]
    active_models = [m for m in model_names if m in transformer_list]
    
    num_models = len(active_models)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))
    
    img = Image.open(image_path).convert('RGB')
    resized_img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    axes[0].imshow(resized_img)
    axes[0].set_title("Original X-Ray")
    axes[0].axis('off')
    
    for i, model_name in enumerate(tqdm(active_models, desc="Generating Rollout")):
        model = load_trained_model(model_name)
        current_size = 384 if model_name == "swin" else config.IMAGE_SIZE
        input_tensor = get_image_tensor(image_path, size=current_size)
        
        recorder = AttentionRecorder(model, model_name)
        
        try:
            with torch.no_grad():
                if model_name == "cnn_transformer":
                    # Bypass TransformerEncoderLayer C++ fast paths completely by manually
                    # looping through the attention computations.
                    x = model.conv4(model.conv3(model.conv2(model.conv1(input_tensor))))
                    x = x.flatten(2).transpose(1, 2)
                    x = model.projection(x) + model.pos_embed
                    
                    attentions = []
                    for layer in model.transformer.layers:
                        # Extract weights by forcing need_weights=True on standard MHA
                        x_attn, weights = layer.self_attn(x, x, x, need_weights=True)
                        attentions.append(weights.detach().unsqueeze(1))
                        
                        # Post-LN progression
                        x = layer.norm1(x + layer.dropout1(x_attn))
                        x_ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                        x = layer.norm2(x + layer.dropout2(x_ff))
                        
                    recorder.attentions = attentions
                else:
                    # Try passing output_attentions=True to trigger matrix return in blocks
                    try:
                        model(input_tensor, output_attentions=True)
                    except TypeError:
                        # Fallback for models whose wrapper doesn't take the arg
                        model(input_tensor)
            
            if not recorder.attentions:
                raise ValueError("No attention matrices captured.")
            
            if model_name == "swin":
                # For Swin, rollout across varying window patches fails in matmul.
                # Average heads and then average across sequence length for the last layer's first window.
                cls_attention = recorder.attentions[-1][0].mean(dim=0).mean(dim=0)
            else:
                rollout = compute_rollout(recorder.attentions)
                current_rollout = rollout[0]
                
                # Extract spatial attention (ignoring CLS if present)
                # Logic: RadDINO (257 tokens), RadJEPA (197/257 tokens), CNN-Trans (49 tokens)
                n_tokens = current_rollout.size(1)
                if n_tokens in [257, 197]: # Typical ViTs with CLS
                    cls_attention = current_rollout[0, 1:]
                elif n_tokens == 49: # CNN-Transformer patches
                    cls_attention = current_rollout.mean(dim=0) # Average attention across all tokens
                else:
                    cls_attention = current_rollout[0, :]
                
            grid_size = int(np.sqrt(cls_attention.size(0)))
            attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
            heatmap = cv2.resize(attention_map, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            
            # Save individual image
            ind_fig, ind_ax = plt.subplots(figsize=(6, 6))
            ind_ax.imshow(heatmap, cmap='jet')
            ind_ax.set_title(f"Rollout: {model_name}")
            ind_ax.axis('off')
            ind_fig.savefig(os.path.join(IMG_DIR, f"{model_name}_rollout.png"), bbox_inches='tight')
            plt.close(ind_fig)

            axes[i+1].imshow(heatmap, cmap='jet')
            axes[i+1].set_title(f"Rollout: {model_name}")
            axes[i+1].axis('off')
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Rollout failed for {model_name}: {e}")
            axes[i+1].text(0.5, 0.5, f'N/A\n(Error)', ha='center', va='center')
            axes[i+1].axis('off')
        finally:
            recorder.remove_hooks()

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "rollout_comparison.png"))
    plt.show()

def run_comparative_lime(image_path, model_names):
    print("Generating LIME comparative view...")
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
    
    img = Image.open(image_path).convert('RGB')
    # Use 224x224 as the base for segmentation
    img_array_224 = np.array(img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))

    explainer = lime_image.LimeImageExplainer()

    for i, model_name in enumerate(tqdm(model_names, desc="Running LIME")):
        current_size = 384 if model_name == "swin" else config.IMAGE_SIZE
        model = load_trained_model(model_name)
        
        # Create model-specific transform
        model_transform = get_preprocess_transform(current_size)

        def batch_predict(images):
            # images come from LIME as numpy arrays
            # We resize each to the model's required input size
            batch = torch.stack([model_transform(Image.fromarray(im)) for im in images]).to(device)
            with torch.no_grad():
                return torch.sigmoid(model(batch)).cpu().numpy()

        explanation = explainer.explain_instance(img_array_224, batch_predict, top_labels=1, num_samples=500)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        
        img_boundary = mark_boundaries(temp/255.0, mask)
        
        # Save individual image
        ind_fig, ind_ax = plt.subplots(figsize=(6, 6))
        ind_ax.imshow(img_boundary)
        ind_ax.set_title(f"LIME: {model_name}")
        ind_ax.axis('off')
        ind_fig.savefig(os.path.join(IMG_DIR, f"{model_name}_lime.png"), bbox_inches='tight')
        plt.close(ind_fig)

        axes[i].imshow(img_boundary)
        axes[i].set_title(f"LIME: {model_name}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "lime_comparison.png"))
    plt.show()

def calculate_metrics(model, image_path, heatmap, model_name, steps=10):
    size = 384 if model_name == "swin" else config.IMAGE_SIZE
    transform = get_preprocess_transform(size)
    img = Image.open(image_path).convert('RGB')
    resized_image = np.array(img.resize((size, size)))
    
    # Scale heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (size, size))
    flat_heatmap = heatmap_resized.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1]
    total_pixels = len(sorted_indices)
    
    mean_color = np.array([0.485, 0.456, 0.406]) * 255
    neutral_image = np.full_like(resized_image, mean_color, dtype=np.uint8)
    
    fractions = np.linspace(0, 1.0, steps + 1)
    deletion_scores = []
    insertion_scores = []

    base_tensor = transform(Image.fromarray(resized_image)).unsqueeze(0).to(device)
    with torch.no_grad():
        base_prob = torch.sigmoid(model(base_tensor))[0]
        target_class = torch.argmax(base_prob).item()

    with torch.no_grad():
        for frac in tqdm(fractions, desc=f"Metrics ({model_name})", leave=False):
            num_pixels = int(frac * total_pixels)
            important_pixels = sorted_indices[:num_pixels]
            
            del_img = resized_image.copy().reshape(-1, 3)
            del_img[important_pixels] = mean_color
            del_tensor = transform(Image.fromarray(del_img.reshape(size, size, 3))).unsqueeze(0).to(device)
            deletion_scores.append(torch.sigmoid(model(del_tensor))[0, target_class].item())
            
            ins_img = neutral_image.copy().reshape(-1, 3)
            ins_img[important_pixels] = resized_image.reshape(-1, 3)[important_pixels]
            ins_tensor = transform(Image.fromarray(ins_img.reshape(size, size, 3))).unsqueeze(0).to(device)
            insertion_scores.append(torch.sigmoid(model(ins_tensor))[0, target_class].item())
            
    return fractions, deletion_scores, insertion_scores

def run_comparative_eval(image_path, model_names):
    print("Generating Insertion/Deletion quantitative comparison...")
    num_models = len(model_names)
    cols = 3
    rows = (num_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    
    for i, model_name in enumerate(tqdm(model_names, desc="Evaluating Models")):
        model = load_trained_model(model_name)
        # In a real scenario, use actual heatmaps. For now using random to match original logic.
        dummy_heatmap = np.random.rand(config.IMAGE_SIZE, config.IMAGE_SIZE)
        
        fracs, del_s, ins_s = calculate_metrics(model, image_path, dummy_heatmap, model_name)
        
        # Save individual plot
        ind_fig, ind_ax = plt.subplots(figsize=(6, 5))
        ind_ax.plot(fracs * 100, del_s, label=f'{model_name} (Del)')
        ind_ax.plot(fracs * 100, ins_s, linestyle='--', label=f'{model_name} (Ins)')
        ind_ax.set_title(f"Faithfulness: {model_name}")
        ind_ax.set_xlabel("Percentage of Pixels Modified")
        ind_ax.set_ylabel("Model Confidence")
        ind_ax.legend()
        ind_ax.grid(True)
        ind_fig.savefig(os.path.join(IMG_DIR, f"{model_name}_faithfulness.png"), bbox_inches='tight')
        plt.close(ind_fig)

        axes[i].plot(fracs * 100, del_s, label=f'{model_name} (Del)')
        axes[i].plot(fracs * 100, ins_s, linestyle='--', label=f'{model_name} (Ins)')
        axes[i].set_title(f"Faithfulness: {model_name}")
        axes[i].set_xlabel("Percentage of Pixels Modified")
        axes[i].set_ylabel("Model Confidence")
        axes[i].legend()
        axes[i].grid(True)
    
    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "faithfulness_comparison.png"))
    plt.show()

def run_comparative_shap(image_path, model_names):
    print("Generating SHAP comparative view...")
    # SHAP can be slow, so we use a reasonable max_evals
    import shap
    
    img = Image.open(image_path).convert('RGB')
    resized_img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    img_array = np.array(resized_img)

    for model_name in tqdm(model_names, desc="Running SHAP"):
        try:
            model = load_trained_model(model_name)
            current_size = 384 if model_name == "swin" else config.IMAGE_SIZE
            model_transform = get_preprocess_transform(current_size)

            def predict_func(images):
                # Ensure images are uint8 for PIL
                batch = torch.stack([model_transform(Image.fromarray(im.astype(np.uint8))) for im in images]).to(device)
                with torch.no_grad():
                    return torch.sigmoid(model(batch)).cpu().numpy()

            # Partition explainer is balanced for speed and quality
            masker = shap.maskers.Image("inpaint_telea", (config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
            explainer = shap.Explainer(predict_func, masker)
            
            # Get prediction for original image to find top class
            base_tensor = model_transform(resized_img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = torch.sigmoid(model(base_tensor)).cpu().numpy()[0]
                top_class = np.argmax(preds)

            # Explain the top class
            shap_values = explainer(img_array.reshape(1, config.IMAGE_SIZE, config.IMAGE_SIZE, 3), 
                                    max_evals=200, batch_size=50, outputs=[top_class])

            # Use shap's built-in image plot
            plt.figure()
            shap.image_plot(shap_values, pixel_values=img_array.reshape(1, config.IMAGE_SIZE, config.IMAGE_SIZE, 3), show=False)
            plt.suptitle(f"SHAP: {model_name} (Class {top_class})", y=0.95)
            plt.savefig(os.path.join(IMG_DIR, f"{model_name}_shap.png"), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"SHAP failed for {model_name}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XAI Comparative Analysis for Chest X-Ray Models")
    parser.add_argument("--image", type=str, default=DEFAULT_TEST_IMAGE, help="Path to the test X-ray image")
    args = parser.parse_args()

    image_path = os.path.join(script_dir, args.image)
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        sys.exit(1)

    # Save original image for reference
    img = Image.open(image_path).convert('RGB')
    img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)).save(os.path.join(IMG_DIR, "original_xray.png"))

    all_models = ["raddino", "radjepa", "swin", "efficientnet", "convnext", "cnn_transformer", "resnet50"]
    # Run comparisons
    # generate_all_gradcam(image_path, all_models)
    # generate_all_rollout(image_path, all_models)
    # run_comparative_lime(image_path, all_models)
    # run_comparative_eval(image_path, all_models)
    run_comparative_shap(image_path, all_models)
