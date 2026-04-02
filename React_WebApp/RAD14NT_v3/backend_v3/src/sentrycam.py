import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class SentryCam:
    def __init__(self, model, target_layer, save_dir):
        self.model = model
        self.save_dir = save_dir
        self.embeddings = []
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Register a PyTorch forward hook on the target classification head.
        # This intercepts the latent vector right before final classification.
        self.hook = target_layer.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        # input[0] contains the latent embeddings (e.g., shape [Batch, 768] for RadDINO)
        latent_vector = input[0].detach().cpu().numpy()
        self.embeddings.append(latent_vector)

    def clear_embeddings(self):
        self.embeddings = []

    def visualize_latent_space(self, dataloader, device, epoch):
        self.model.eval()
        self.clear_embeddings()
        
        all_labels = []
        
        # Process a small subset (e.g., 500 images) to keep t-SNE fast and readable
        max_samples = 500
        samples_collected = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                if samples_collected >= max_samples:
                    break
                images = images.to(device)
                
                # The forward pass triggers the hook_fn automatically
                _ = self.model(images)
                
                all_labels.append(labels.cpu().numpy())
                samples_collected += images.size(0)
                
        # Stack all intercepted embeddings and labels
        latent_space = np.vstack(self.embeddings)[:max_samples]
        targets = np.vstack(all_labels)[:max_samples]
        
        # Flatten any extra spatial or sequence dimensions to ensure shape is (n_samples, n_features)
        if latent_space.ndim > 2:
            latent_space = latent_space.reshape(latent_space.shape[0], -1)
        
        # Reduce the high-dimensional ViT space down to 2D for plotting
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        latent_2d = tsne.fit_transform(latent_space)
        
        plt.figure(figsize=(12, 10))
        
        # Isolate the dominant "No Finding" class (rows where all 14 labels are exactly 0)
        is_healthy = np.sum(targets, axis=1) == 0
        
        # Plot healthy patients as the baseline background
        plt.scatter(
            latent_2d[is_healthy, 0], 
            latent_2d[is_healthy, 1], 
            c='lightgray', label='No Finding (Healthy)', alpha=0.5, edgecolors='w', s=60
        )
        
        # Plot the patients with pathologies
        plt.scatter(
            latent_2d[~is_healthy, 0], 
            latent_2d[~is_healthy, 1], 
            c='crimson', label='Pathology Present', alpha=0.8, edgecolors='w', s=60
        )
        
        plt.title(f'SentryCam Latent Space Evolution - Epoch {epoch}')
        plt.xlabel('Latent Dimension 1 (t-SNE)')
        plt.ylabel('Latent Dimension 2 (t-SNE)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        save_path = os.path.join(self.save_dir, f'latent_epoch_{epoch}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"SentryCam generated visualization: {save_path}")
        
    def close(self):
        # Clean up and remove the hook when training is completely finished
        self.hook.remove()