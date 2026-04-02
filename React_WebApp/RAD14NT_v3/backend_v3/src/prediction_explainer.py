import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
from PIL import Image
from model import get_model
from config import IMAGE_SIZE, MODEL_NAME

def run_shap(image_path, model_weights_path, background_images_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    background_images_tensor = background_images_tensor.to(device)

    explainer = shap.GradientExplainer(model, background_images_tensor)
    shap_values, indexes = explainer.shap_values(input_tensor, ranked_outputs=1)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(input_tensor.cpu().numpy(), 1, -1), 1, 2)
    
    shap.image_plot(shap_numpy, -test_numpy, show=False)
    plt.savefig(f'shap_output_{MODEL_NAME}.png')

def run_lime(image_path, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def batch_predict(images):
        model.eval()
        batch = torch.stack(tuple(transform(Image.fromarray(i)) for i in images), dim=0)
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array, 
        batch_predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    img_boundry = mark_boundaries(temp/255.0, mask)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_boundry)
    plt.title(f'LIME Explanation ({MODEL_NAME})')
    plt.axis('off')
    plt.savefig(f'lime_output_{MODEL_NAME}.png')
    plt.show()

if __name__ == "__main__":
    # Example Usage:
    run_lime("../dataset/sample_xray.png", "path_to_saved_model.pth")