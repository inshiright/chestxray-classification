import json
import os

notebook_path = r"c:\Users\Rald999\Documents\GitHub\chestxray-classification\notebooks\xai_explanations.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Patch reshape_transform cell (id 'paths')
for cell in nb['cells']:
    if cell.get('id') == 'paths':
        cell['source'] = [
            "# --- configuration ---\n",
            "MODEL_WEIGHTS = {\n",
            "    \"raddino\": \"../model_checkpoints/raddino_best_model.pth\",\n",
            "    \"radjepa\": \"../model_checkpoints/radjepa_best_model.pth\",\n",
            "    \"swin\": \"../model_checkpoints/swin_best_model.pth\",\n",
            "    \"efficientnet\": \"../model_checkpoints/efficientnet_best_model.pth\",\n",
            "    \"convnext\": \"../model_checkpoints/convnext_best_model.pth\",\n",
            "    \"cnn_transformer\": \"../model_checkpoints/cnn_transformer_best_model.pth\",\n",
            "    \"resnet50\": \"../model_checkpoints/resnet50_best_model.pth\"\n",
            "}\n",
            "\n",
            "TEST_IMAGES = [\n",
            "    \"../dataset/00000001_002.png\",\n",
            "    \"../dataset/00000003_003.png\",\n",
            "    \"../dataset/00000005_007.png\",\n",
            "    \"../dataset/00000008_001.png\",\n",
            "    \"../dataset/00000008_002.png\"\n",
            "]\n",
            "\n",
            "# Utility functions for Grad-CAM with Transformers\n",
            "def reshape_transform_vit(tensor):\n",
            "    n_tokens = tensor.size(1)\n",
            "    if int(np.sqrt(n_tokens - 1))**2 == n_tokens - 1:\n",
            "        grid_size = int(np.sqrt(n_tokens - 1))\n",
            "        result = tensor[:, 1:, :].reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))\n",
            "    else:\n",
            "        grid_size = int(np.sqrt(n_tokens))\n",
            "        result = tensor.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))\n",
            "    \n",
            "    result = result.permute(0, 3, 1, 2)\n",
            "    return result\n",
            "\n",
            "def reshape_transform_h_w_c(tensor):\n",
            "    return tensor.permute(0, 3, 1, 2)"
        ]

    if cell.get('id') == 'gradcam_impl':
        cell['source'] = [
            "def generate_all_gradcam(image_path, model_names):\n",
            "    num_models = len(model_names)\n",
            "    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))\n",
            "    \n",
            "    img = Image.open(image_path).convert('RGB')\n",
            "    axes[0].imshow(img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))\n",
            "    axes[0].set_title(\"Original X-Ray\")\n",
            "    axes[0].axis('off')\n",
            "    \n",
            "    input_tensor = get_image_tensor(image_path)\n",
            "    rgb_img = np.float32(img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))) / 255\n",
            "\n",
            "    for i, model_name in enumerate(tqdm(model_names, desc=\"Grad-CAM Models\")):\n",
            "        # Handle models with different input size requirements\n",
            "        current_size = 384 if model_name == \"swin\" else config.IMAGE_SIZE\n",
            "        model = load_trained_model(model_name)\n",
            "        input_tensor = get_image_tensor(image_path, size=current_size)\n",
            "        rgb_img_model = np.float32(img.resize((current_size, current_size))) / 255\n",
            "        \n",
            "        reshape_transform = None\n",
            "        if model_name == \"efficientnet\":\n",
            "            target_layers = [model.model.conv_head]\n",
            "        elif model_name == \"convnext\":\n",
            "            target_layers = [model.model.stages[-1].blocks[-1]]\n",
            "        elif model_name == \"resnet50\":\n",
            "            target_layers = [model.backbone.layer4[-1]] \n",
            "        elif model_name == \"raddino\":\n",
            "            target_layers = [model.encoder.encoder.layer[-1].norm1]\n",
            "            reshape_transform = reshape_transform_vit\n",
            "        elif model_name == \"radjepa\":\n",
            "            target_layers = [model.encoder.model.blocks[-1].norm1]\n",
            "            reshape_transform = reshape_transform_vit\n",
            "        elif model_name == \"swin\":\n",
            "            target_layers = [model.model.layers[-1].blocks[-1]]\n",
            "            reshape_transform = reshape_transform_h_w_c\n",
            "        elif model_name == \"cnn_transformer\":\n",
            "            target_layers = [model.conv4[-1]]\n",
            "        else:\n",
            "            continue\n",
            "\n",
            "        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)\n",
            "        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]\n",
            "        cam_image = show_cam_on_image(rgb_img_model, grayscale_cam, use_rgb=True)\n",
            "        \n",
            "        axes[i+1].imshow(cam_image)\n",
            "        axes[i+1].set_title(f\"Grad-CAM: {model_name}\")\n",
            "        axes[i+1].axis('off')\n",
            "\n",
            "    plt.tight_layout()\n",
            "    plt.show()"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
