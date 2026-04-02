"""
Flask API server for chest x-ray analysis.
Provides: predictions, GradCAM, LIME, SHAP, Attention Rollout, Blackout/Insertion-Deletion

Usage:
    python api.py --weights checkpoints/efficientnet_best_model.pth
    python api.py --weights checkpoints/raddino_best_model.pth --model raddino

Then from the frontend, set VITE_API_URL=http://localhost:5000
"""

import os
import sys
import io
import base64
import argparse
import traceback

root_path = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(root_path, "src")
for p in (root_path, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms
from PIL import Image
from sklearn.metrics import auc as sklearn_auc

# ── Config ────────────────────────────────────────────────────────────────────
DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(DISEASE_CLASSES)
IMAGE_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _model_input_size():
    # RadJEPA internally resizes to 224x224, so we feed it 224 at the API level too.
    # RadDINO uses the full 518x518 native resolution.
    if MODEL_NAME == "raddino":
        return 518
    elif MODEL_NAME == "swin":
        return 384
    return IMAGE_SIZE

def _make_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

app = Flask(__name__)
CORS(app)  # allow requests from the Vite dev server

SAMPLES_DIR = os.path.join(root_path, "static", "samples")


@app.route("/samples/<filename>")
def serve_sample(filename):
    from flask import send_from_directory
    return send_from_directory(SAMPLES_DIR, filename)

# ── Globals (loaded once on startup) ─────────────────────────────────────────
MODEL = None
DEVICE = None
MODEL_NAME = None
CPU_MODEL = None   # persistent CPU copy used exclusively by SHAP — avoids deepcopy on every request


# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_from_request() -> Image.Image:
    """Read the uploaded image from the current Flask request."""
    file = request.files.get("image")
    if file is None:
        raise ValueError("No 'image' field in request.")
    img = Image.open(file.stream).convert("RGB")
    return img


def img_to_numpy(pil_img: Image.Image, size: int = None) -> np.ndarray:
    """Resize PIL image → uint8 numpy (H, W, 3)."""
    s = size or _model_input_size()
    return np.array(pil_img.resize((s, s)))


def numpy_to_tensor(np_img: np.ndarray, size: int = None) -> torch.Tensor:
    """uint8 numpy (H,W,3) → normalised float tensor (1,3,H,W)."""
    s = size or _model_input_size()
    t = _make_transform(s)
    return t(Image.fromarray(np_img)).unsqueeze(0).to(DEVICE)


def fig_to_b64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def numpy_img_to_b64(np_img: np.ndarray) -> str:
    """Encode a uint8 numpy image (H,W,3) as a base64 PNG."""
    success, buf = cv2.imencode(".png", cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("cv2.imencode failed")
    encoded = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ── Explainability helpers ────────────────────────────────────────────────────

def _run_radjepa_forward(tensor: torch.Tensor):
    """
    Run RadJEPA forward pass. RadJEPA's encoder is a plain ViT loaded via
    AutoModel — it does NOT expose HuggingFace config.output_attentions or
    pooler_output. We call forward() directly and pool the CLS token.

    Returns logits tensor (not sigmoid'd).
    """
    outputs = MODEL.encoder(tensor)
    # Handle both HuggingFace BaseOutput and plain tensor/tuple returns
    if hasattr(outputs, "last_hidden_state"):
        features = outputs.last_hidden_state
    elif isinstance(outputs, (list, tuple)):
        features = outputs[0]
    else:
        features = outputs

    if features.dim() == 3:
        cls_token = features[:, 0, :]
    else:
        cls_token = features

    return MODEL.classifier(cls_token)


def _get_attention_heatmap(input_tensor: torch.Tensor):
    """
    Compute attention-rollout heatmap.
    Works for RadDINO (HuggingFace ViT with .encoder.config) only.
    RadJEPA uses _get_radjepa_attention_heatmap() instead.
    Returns (heatmap np float32 [H,W], probs np float32 [C]).
    """
    MODEL.encoder.config.output_attentions = True
    MODEL.encoder.config.return_dict = True

    outputs = MODEL.encoder(input_tensor, output_attentions=True, return_dict=True)
    attentions = outputs.attentions

    cls_token = outputs.pooler_output
    logits = MODEL.classifier(cls_token)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    # Attention rollout
    result = torch.eye(attentions[0].size(-1)).to(DEVICE)
    for attn in attentions:
        fused = attn.mean(dim=1)
        fused += torch.eye(fused.size(-1)).to(DEVICE)
        fused = fused / fused.sum(dim=-1, keepdim=True)
        result = torch.matmul(fused, result)

    cls_attn = result[0, 0, 1:]
    grid = int(np.sqrt(cls_attn.size(0)))
    attn_map = cls_attn.reshape(grid, grid).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    heatmap = cv2.resize(attn_map, (IMAGE_SIZE, IMAGE_SIZE))
    return heatmap.astype(np.float32), probs


def _get_radjepa_attention_heatmap(input_tensor: torch.Tensor):
    """
    Attention rollout for RadJEPA — which loads via AutoModel and stores its
    ViT blocks in MODEL.encoder.model.blocks (a plain nn.ModuleList).
    We register forward hooks to collect attention weights directly.

    Returns (heatmap np float32 [H,W], probs np float32 [C]).
    """
    attention_weights = []

    def _hook(module, input, output):
        # timm ViT attention: output is typically (x,) or (x, attn_weights)
        # depending on whether attn_drop returns weights.
        # We access the weights by hooking the softmax output inside Attention.
        pass

    # Collect attn weights by hooking into each block's attn.softmax
    hooks = []
    attn_maps = []

    def make_hook(attn_maps_list):
        def hook_fn(module, input, output):
            # output from F.softmax: shape [B, heads, N, N]
            attn_maps_list.append(output.detach())
        return hook_fn

    for block in MODEL.encoder.model.blocks:
        # timm ViT Block: block.attn.attn_drop is after softmax
        # We hook the attn module's forward to capture after softmax
        hook = block.attn.attn_drop.register_forward_hook(make_hook(attn_maps))
        hooks.append(hook)

    try:
        with torch.no_grad():
            logits = _run_radjepa_forward(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
    finally:
        for h in hooks:
            h.remove()

    if not attn_maps:
        # Fallback: return uniform heatmap if hooks captured nothing
        heatmap = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        return heatmap, probs

    # Attention rollout
    n_tokens = attn_maps[0].size(-1)
    result = torch.eye(n_tokens).to(DEVICE)
    for attn in attn_maps:
        # attn shape: [B, heads, N, N]
        fused = attn[0].mean(dim=0)  # [N, N]
        fused = fused + torch.eye(n_tokens).to(DEVICE)
        fused = fused / fused.sum(dim=-1, keepdim=True)
        result = torch.matmul(fused, result)

    cls_attn = result[0, 1:]  # [N-1] — patch tokens
    grid = int(np.sqrt(cls_attn.size(0)))
    attn_map = cls_attn.reshape(grid, grid).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    heatmap = cv2.resize(attn_map, (IMAGE_SIZE, IMAGE_SIZE))
    return heatmap.astype(np.float32), probs


def _get_any_attention_heatmap(input_tensor: torch.Tensor):
    """Dispatch attention rollout to the correct implementation."""
    if MODEL_NAME == "raddino":
        return _get_attention_heatmap(input_tensor)
    elif MODEL_NAME == "radjepa":
        return _get_radjepa_attention_heatmap(input_tensor)
    else:
        raise ValueError(f"Attention rollout not available for {MODEL_NAME}")


def _is_vit_model():
    return MODEL_NAME in ("raddino", "radjepa")


def _get_gradcam_target_layers():
    if MODEL_NAME == "efficientnet":
        # timm EfficientNet uses .blocks, not .features (that's torchvision's convention)
        return [MODEL.model.blocks[-1]], None
    elif MODEL_NAME == "convnext":
        # timm ConvNeXt uses .stages, not .features
        return [MODEL.model.stages[-1]], None
    elif MODEL_NAME == "resnet50":
        # torchvision ResNet: backbone.layer4 is the final residual stage
        return [MODEL.backbone.layer4[-1]], None
    elif MODEL_NAME == "swin":
        # Swin norm1 hook captures [B, H, W, C] — already spatial, just permute to [B, C, H, W]
        def reshape_swin(tensor):
            return tensor.permute(0, 3, 1, 2)
        return [MODEL.model.layers[-1].blocks[-1].norm1], reshape_swin
    elif MODEL_NAME == "raddino":
        def reshape_vit(tensor):
            gs = int((tensor.size(1) - 1) ** 0.5)
            r = tensor[:, 1:, :].reshape(tensor.size(0), gs, gs, tensor.size(2))
            return r.permute(0, 3, 1, 2)
        return [MODEL.encoder.encoder.layer[-1].layernorm_before], reshape_vit
    elif MODEL_NAME == "radjepa":
        def reshape_vit(tensor):
            gs = int((tensor.size(1) - 1) ** 0.5)
            r = tensor[:, 1:, :].reshape(tensor.size(0), gs, gs, tensor.size(2))
            return r.permute(0, 3, 1, 2)
        return [MODEL.encoder.model.blocks[-1].norm1], reshape_vit
    else:
        raise ValueError(f"No GradCAM config for model: {MODEL_NAME}")


def _infer_probs(tensor: torch.Tensor) -> np.ndarray:
    """Run inference and return sigmoid probabilities as numpy array."""
    with torch.no_grad():
        if MODEL_NAME == "raddino":
            MODEL.encoder.config.output_attentions = False
            outputs = MODEL.encoder(tensor, return_dict=True)
            cls_token = outputs.pooler_output
            logits = MODEL.classifier(cls_token)
        elif MODEL_NAME == "radjepa":
            logits = _run_radjepa_forward(tensor)
        else:
            logits = MODEL(tensor)
        return torch.sigmoid(logits)[0].cpu().numpy()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": str(DEVICE)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Returns per-disease confidence scores + a GradCAM/Attention heatmap for the top predicted class.
    Response: { predictions: [{disease, confidence}], topDisease, topConfidence, heatmapUrl }
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        pil_img = pil_from_request()

        # ── Determine input size ──────────────────────────────────────────────
        size = _model_input_size()
        np_img = np.array(pil_img.convert("RGB").resize((size, size)))
        rgb_float = np.float32(np_img) / 255.0

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        # NOTE: tensor is created OUTSIDE no_grad so GradCAM can compute gradients later
        tensor = _transform(np_img).unsqueeze(0).to(DEVICE)

        # ── Inference ─────────────────────────────────────────────────────────
        probs = _infer_probs(tensor)

        predictions = [
            {"disease": DISEASE_CLASSES[i], "confidence": float(probs[i])}
            for i in range(NUM_CLASSES)
        ]
        top_idx = int(np.argmax(probs))

        # ── Heatmap ───────────────────────────────────────────────────────────
        heatmap_b64 = None
        try:
            if MODEL_NAME == "efficientnet":
                target_layers = [MODEL.model.blocks[-1]]
                cam = GradCAM(model=MODEL, target_layers=target_layers)
                MODEL.eval()
                with torch.enable_grad():
                    t = _transform(np_img).unsqueeze(0).to(DEVICE)
                    grayscale_cam = cam(input_tensor=t, targets=None)[0]

            elif MODEL_NAME == "resnet50":
                target_layers = [MODEL.backbone.layer4[-1]]
                cam = GradCAM(model=MODEL, target_layers=target_layers)
                MODEL.eval()
                with torch.enable_grad():
                    t = _transform(np_img).unsqueeze(0).to(DEVICE)
                    grayscale_cam = cam(input_tensor=t, targets=None)[0]

            elif MODEL_NAME == "raddino":
                class _ViTWrapper(torch.nn.Module):
                    def __init__(self, m): super().__init__(); self.model = m
                    def forward(self, x):
                        outputs = self.model.encoder(x)
                        tokens = (outputs[0] if isinstance(outputs, (list, tuple))
                                  else outputs.get("last_hidden_state",
                                                   next(iter(outputs.values())))
                                  if isinstance(outputs, dict) else outputs)
                        return tokens[:, 0, :]

                wrapped = _ViTWrapper(MODEL)

                def reshape_vit(t, h=37, w=37):
                    r = t[:, 1:, :].reshape(t.size(0), h, w, t.size(2))
                    return r.transpose(2, 3).transpose(1, 2)

                tl = wrapped.model.encoder.encoder.layer[-1].norm1
                cam = GradCAM(model=wrapped, target_layers=[tl],
                              reshape_transform=reshape_vit)
                wrapped.eval()
                with torch.enable_grad():
                    t = _transform(np_img).unsqueeze(0).to(DEVICE)
                    grayscale_cam = cam(input_tensor=t, targets=None)[0]

            elif MODEL_NAME == "radjepa":
                # RadJEPA: use attention rollout (more reliable than GradCAM for this model)
                with torch.no_grad():
                    grayscale_cam, _ = _get_radjepa_attention_heatmap(tensor)

            else:
                target_layers, reshape_transform = _get_gradcam_target_layers()
                cam = GradCAM(model=MODEL, target_layers=target_layers,
                              reshape_transform=reshape_transform)
                MODEL.eval()
                with torch.enable_grad():
                    t = _transform(np_img).unsqueeze(0).to(DEVICE)
                    grayscale_cam = cam(input_tensor=t, targets=None)[0]

            # Resize heatmap to match np_img dimensions for overlay
            if grayscale_cam.shape[:2] != np_img.shape[:2]:
                grayscale_cam = cv2.resize(grayscale_cam, (np_img.shape[1], np_img.shape[0]))
            cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
            heatmap_b64 = numpy_img_to_b64(cam_image)
        except Exception as cam_err:
            print(f"Heatmap in /predict failed (non-fatal): {cam_err}")
            traceback.print_exc()

        return jsonify({
            "predictions": predictions,
            "topDisease": DISEASE_CLASSES[top_idx],
            "topConfidence": float(probs[top_idx]),
            "heatmapUrl": heatmap_b64,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/explain/gradcam", methods=["POST"])
def explain_gradcam():
    """
    Returns a GradCAM heatmap overlay as base64 PNG.

    EfficientNet B0 (timm): target layer is model.model.blocks[-1], no reshape needed.
    ViT models (RadDINO / RadJEPA): uses a wrapper so patch token gradients are retained,
    with a 37x37 reshape transform for the 518-token grid.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        pil_img = pil_from_request()

        if MODEL_NAME == "efficientnet":
            # ── EfficientNet B0 ───────────────────────────────────────────────
            np_img  = img_to_numpy(pil_img)           # 224×224
            rgb_float = np.float32(np_img) / 255.0
            tensor = numpy_to_tensor(np_img)

            target_layers = [MODEL.model.blocks[-1]]
            cam = GradCAM(model=MODEL, target_layers=target_layers)

            MODEL.eval()
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=tensor, targets=None)[0]

        elif MODEL_NAME == "raddino":
            # ── RadDINO ─────────────────────────────────────────────────────
            # 518×518 input; tokens are [CLS, patch_1 … patch_1369] → 37×37 grid
            size = _model_input_size()

            np_img = np.array(pil_img.resize((size, size)))
            rgb_float = np.float32(np_img) / 255.0

            _transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            tensor = _transform(np_img).unsqueeze(0).to(DEVICE)

            class _ViTWrapper(torch.nn.Module):
                def __init__(self, m): super().__init__(); self.model = m
                def forward(self, x):
                    outputs = self.model.encoder(x)
                    tokens = (outputs[0] if isinstance(outputs, (list, tuple))
                              else outputs.get("last_hidden_state",
                                               next(iter(outputs.values())))
                              if isinstance(outputs, dict) else outputs)
                    return tokens[:, 0, :]   # CLS token

            wrapped = _ViTWrapper(MODEL)

            def reshape_vit(tensor, h=37, w=37):
                result = tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(2))
                return result.transpose(2, 3).transpose(1, 2)

            target_layers = [wrapped.model.encoder.encoder.layer[-1].norm1]
            cam = GradCAM(model=wrapped, target_layers=target_layers,
                          reshape_transform=reshape_vit)

            wrapped.eval()
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=tensor, targets=None)[0]

        elif MODEL_NAME == "radjepa":
            # ── RadJEPA — use Attention Rollout (GradCAM not applicable) ────
            np_img = img_to_numpy(pil_img)
            rgb_float = np.float32(np_img) / 255.0
            tensor = numpy_to_tensor(np_img)

            with torch.no_grad():
                grayscale_cam, _ = _get_radjepa_attention_heatmap(tensor)

        else:
            # ── Generic CNN fallback (ConvNeXt, Swin, ResNet50) ──────────────
            np_img = img_to_numpy(pil_img)
            rgb_float = np.float32(np_img) / 255.0
            tensor = numpy_to_tensor(np_img)
            target_layers, reshape_transform = _get_gradcam_target_layers()
            cam = GradCAM(model=MODEL, target_layers=target_layers,
                          reshape_transform=reshape_transform)
            MODEL.eval()
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=tensor, targets=None)[0]

        if grayscale_cam.shape[:2] != np_img.shape[:2]:
            grayscale_cam = cv2.resize(grayscale_cam, (np_img.shape[1], np_img.shape[0]))
        cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        return jsonify({"image": numpy_img_to_b64(cam_image)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/explain/attention_rollout", methods=["POST"])
def explain_attention_rollout():
    """
    Attention Rollout for ViT-based models (RadDINO, RadJEPA).
    Returns overlay image as base64 PNG.
    """
    try:
        if not _is_vit_model():
            return jsonify({"error": "Attention Rollout is only available for ViT-based models (raddino, radjepa)."}), 400

        pil_img = pil_from_request()
        np_img = img_to_numpy(pil_img)
        tensor = numpy_to_tensor(np_img)

        with torch.no_grad():
            heatmap, _ = _get_any_attention_heatmap(tensor)

        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(np_img, 0.5, heatmap_color, 0.5, 0)

        # Side-by-side figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(np_img); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Attention Map"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
        plt.tight_layout()

        return jsonify({"image": fig_to_b64(fig), "overlay": numpy_img_to_b64(overlay)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/explain/lime", methods=["POST"])
def explain_lime():
    """
    LIME image explanation. Returns the boundary-marked image as base64 PNG.
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        pil_img = pil_from_request()
        np_img = img_to_numpy(pil_img)

        def batch_predict(images):
            MODEL.eval()
            t = _make_transform(_model_input_size())
            batch = torch.stack([t(Image.fromarray(i)) for i in images]).to(DEVICE)
            with torch.no_grad():
                if MODEL_NAME == "raddino":
                    MODEL.encoder.config.output_attentions = False
                    outputs = MODEL.encoder(batch, return_dict=True)
                    logits = MODEL.classifier(outputs.pooler_output)
                elif MODEL_NAME == "radjepa":
                    outputs = MODEL.encoder(batch)
                    if hasattr(outputs, "last_hidden_state"):
                        features = outputs.last_hidden_state
                    elif isinstance(outputs, (list, tuple)):
                        features = outputs[0]
                    else:
                        features = outputs
                    cls_token = features[:, 0, :] if features.dim() == 3 else features
                    logits = MODEL.classifier(cls_token)
                else:
                    logits = MODEL(batch)
                probs = torch.sigmoid(logits)
            return probs.cpu().numpy()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np_img, batch_predict,
            top_labels=1, hide_color=0, num_samples=300,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
        )
        img_boundary = mark_boundaries(temp / 255.0, mask)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_boundary)
        ax.set_title(f"LIME — {DISEASE_CLASSES[explanation.top_labels[0]]}")
        ax.axis("off")

        return jsonify({"image": fig_to_b64(fig)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/explain/shap", methods=["POST"])
def explain_shap():
    """
    SHAP GradientExplainer.
    Uses a black (zero) tensor as the baseline — same approach as the notebook.
    Shows the top predicted class label on the plot.

    Not supported for Swin Transformer (gradient issues with shifted windows).
    RadDINO/RadJEPA run on CPU via the persistent CPU_MODEL copy to avoid GPU OOM.
    """
    try:
        if MODEL_NAME == "swin":
            return jsonify({
                "error": "SHAP is not supported for Swin Transformer. "
                         "Use GradCAM or LIME instead."
            }), 400

        import shap
        import copy

        pil_img = pil_from_request()

        size = _model_input_size()

        # SHAP input size: cap at 224px to keep memory low on free servers.
        # Attribution maps are spatially smooth — computing at 224px then
        # upsampling to native size for display is visually equivalent.
        SHAP_SIZE = min(size, 224)
        shap_transform = _make_transform(SHAP_SIZE)
        tensor_cpu = shap_transform(pil_img.convert("RGB")).unsqueeze(0)  # [1,3,S,S] on CPU

        # Wrap CPU_MODEL in a thin adapter that resizes input to native size
        # so the model never sees a wrong-resolution tensor
        class _ResizeWrapper(torch.nn.Module):
            def __init__(self, m, native):
                super().__init__()
                self.m = m
                self.native = native
            def forward(self, x):
                if x.shape[-1] != self.native:
                    x = torch.nn.functional.interpolate(
                        x, size=(self.native, self.native),
                        mode="bilinear", align_corners=False
                    )
                return self.m(x)

        wrapped_cpu = _ResizeWrapper(CPU_MODEL, size)
        wrapped_cpu.eval()

        background_cpu = torch.zeros(1, 3, SHAP_SIZE, SHAP_SIZE)

        # GradientExplainer on CPU — avoids GPU memory pressure
        explainer = shap.GradientExplainer(wrapped_cpu, background_cpu)
        shap_values, indexes = explainer.shap_values(
            tensor_cpu, ranked_outputs=1, nsamples=20
        )

        top_class_idx = indexes[0][0]
        if hasattr(top_class_idx, "item"):
            top_class_idx = top_class_idx.item()
        top_label = DISEASE_CLASSES[top_class_idx]

        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy  = np.swapaxes(np.swapaxes(tensor_cpu.numpy(), 1, -1), 1, 2)

        del wrapped_cpu, background_cpu, explainer

        # shap.image_plot() creates its own figure internally — do NOT pre-create
        # one with plt.figure() or fig_to_b64 will encode that empty figure instead.
        shap.image_plot(shap_numpy, -test_numpy,
                        labels=[[top_label]], show=False)
        plt.suptitle(f"SHAP — top predicted: {top_label}", fontsize=10, y=1.01)

        # Grab the figure shap.image_plot actually drew into
        fig = plt.gcf()

        return jsonify({"image": fig_to_b64(fig), "topClass": top_label})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/explain/blackout", methods=["POST"])
def explain_blackout():
    """
    Blackout / Insertion-Deletion test (pixel-level faithfulness metric).
    Works for all models. Returns the insertion/deletion curve as base64 PNG,
    plus the AUC scores.

    For CNN models: uses GradCAM heatmap to rank pixels.
    For ViT models: uses Attention Rollout heatmap to rank pixels.
    """
    try:
        steps = int(request.form.get("steps", 10))

        pil_img = pil_from_request()
        np_img = img_to_numpy(pil_img)
        tensor = numpy_to_tensor(np_img)

        # ── Get heatmap & base probs ──────────────────────────────────────────
        if _is_vit_model():
            with torch.no_grad():
                heatmap, base_probs = _get_any_attention_heatmap(tensor)
        else:
            # Use GradCAM heatmap for CNN models
            from pytorch_grad_cam import GradCAM
            target_layers, reshape_transform = _get_gradcam_target_layers()
            cam = GradCAM(model=MODEL, target_layers=target_layers, reshape_transform=reshape_transform)
            with torch.enable_grad():
                heatmap = cam(input_tensor=tensor, targets=None)[0].astype(np.float32)

            with torch.no_grad():
                logits = MODEL(tensor)
                base_probs = torch.sigmoid(logits)[0].cpu().numpy()

        target_idx = int(np.argmax(base_probs))
        base_conf = float(base_probs[target_idx])

        # ── Pixel importance sort ─────────────────────────────────────────────
        sorted_indices = np.argsort(heatmap.flatten())[::-1]
        total_pixels = len(sorted_indices)
        mean_color = (np.array([0.485, 0.456, 0.406]) * 255).astype(np.uint8)
        neutral_image = np.full_like(np_img, mean_color)

        deletion_scores, insertion_scores = [], []
        fractions = np.linspace(0, 1.0, steps + 1)

        def infer_confidence(img_arr):
            t = numpy_to_tensor(img_arr)
            return float(_infer_probs(t)[target_idx])

        for frac in fractions:
            n = int(frac * total_pixels)
            pixels = sorted_indices[:n]

            del_img = np_img.copy().reshape(-1, 3)
            if n > 0:
                del_img[pixels] = neutral_image.reshape(-1, 3)[0]
            deletion_scores.append(infer_confidence(del_img.reshape(_model_input_size(), _model_input_size(), 3)))

            ins_img = neutral_image.copy().reshape(-1, 3)
            if n > 0:
                ins_img[pixels] = np_img.reshape(-1, 3)[pixels]
            insertion_scores.append(infer_confidence(ins_img.reshape(_model_input_size(), _model_input_size(), 3)))

        del_auc = float(sklearn_auc(fractions, deletion_scores))
        ins_auc = float(sklearn_auc(fractions, insertion_scores))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fractions * 100, deletion_scores,
                label=f"Deletion (AUC: {del_auc:.3f})", color="red", marker="o")
        ax.plot(fractions * 100, insertion_scores,
                label=f"Insertion (AUC: {ins_auc:.3f})", color="blue", marker="s")
        ax.axhline(y=base_conf, color="gray", linestyle="--", alpha=0.5,
                   label=f"Baseline confidence ({base_conf:.3f})")
        ax.set_title(f"Insertion / Deletion — {DISEASE_CLASSES[target_idx]}")
        ax.set_xlabel("% Pixels Modified")
        ax.set_ylabel("Model Confidence")
        ax.legend()
        ax.grid(True)

        return jsonify({
            "image": fig_to_b64(fig),
            "targetDisease": DISEASE_CLASSES[target_idx],
            "baseConfidence": base_conf,
            "deletionAUC": del_auc,
            "insertionAUC": ins_auc,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/explain/all", methods=["POST"])
def explain_all():
    """
    Run all available explainability methods in one shot.
    Returns a combined JSON with all results.
    Slow — intended for when the user clicks "Run Full Analysis".
    """
    results = {}

    # Always run predict
    with app.test_request_context():
        pass  # just for IDE

    pil_img = pil_from_request()

    def _sub_request(endpoint_fn):
        """Re-use the helper functions directly to avoid HTTP overhead."""
        pass

    # We'll delegate to the individual route functions by re-reading the file
    # and passing a BytesIO fake file — simpler: just call the logic directly.

    try:
        np_img = img_to_numpy(pil_img)
        tensor = numpy_to_tensor(np_img)

        # ── Predictions ───────────────────────────────────────────────────────
        probs = _infer_probs(tensor)

        results["predictions"] = [
            {"disease": DISEASE_CLASSES[i], "confidence": float(probs[i])}
            for i in range(NUM_CLASSES)
        ]
        top_idx = int(np.argmax(probs))
        results["topDisease"] = DISEASE_CLASSES[top_idx]
        results["topConfidence"] = float(probs[top_idx])

        # ── Heatmap (shared between rollout/blackout) ─────────────────────────
        if _is_vit_model():
            with torch.no_grad():
                heatmap, _ = _get_any_attention_heatmap(tensor)

            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(np_img, 0.5, heatmap_color, 0.5, 0)
            results["attentionRollout"] = numpy_img_to_b64(overlay)
        else:
            try:
                from pytorch_grad_cam import GradCAM
                from pytorch_grad_cam.utils.image import show_cam_on_image
                target_layers, reshape_transform = _get_gradcam_target_layers()
                cam = GradCAM(model=MODEL, target_layers=target_layers, reshape_transform=reshape_transform)
                rgb_float = np.float32(np_img) / 255.0
                grayscale_cam = cam(input_tensor=tensor, targets=None)[0]
                heatmap = grayscale_cam.astype(np.float32)
                cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
                results["gradcam"] = numpy_img_to_b64(cam_image)
            except Exception as e:
                results["gradcam"] = None
                results["gradcamError"] = str(e)

        # ── Blackout ──────────────────────────────────────────────────────────
        try:
            sorted_indices = np.argsort(heatmap.flatten())[::-1]
            total_pixels = len(sorted_indices)
            mean_color = (np.array([0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            neutral_image = np.full_like(np_img, mean_color)
            steps = 10
            fractions = np.linspace(0, 1.0, steps + 1)
            base_conf = float(probs[top_idx])

            deletion_scores, insertion_scores = [], []

            def infer_conf(img_arr):
                t = numpy_to_tensor(img_arr)
                return float(_infer_probs(t)[top_idx])

            for frac in fractions:
                n = int(frac * total_pixels)
                pixels = sorted_indices[:n]
                d = np_img.copy().reshape(-1, 3)
                if n > 0:
                    d[pixels] = neutral_image.reshape(-1, 3)[0]
                deletion_scores.append(infer_conf(d.reshape(_model_input_size(), _model_input_size(), 3)))
                ins = neutral_image.copy().reshape(-1, 3)
                if n > 0:
                    ins[pixels] = np_img.reshape(-1, 3)[pixels]
                insertion_scores.append(infer_conf(ins.reshape(_model_input_size(), _model_input_size(), 3)))

            del_auc = float(sklearn_auc(fractions, deletion_scores))
            ins_auc = float(sklearn_auc(fractions, insertion_scores))

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(fractions * 100, deletion_scores,
                    label=f"Deletion (AUC: {del_auc:.3f})", color="red", marker="o")
            ax.plot(fractions * 100, insertion_scores,
                    label=f"Insertion (AUC: {ins_auc:.3f})", color="blue", marker="s")
            ax.axhline(y=base_conf, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(f"Insertion / Deletion — {DISEASE_CLASSES[top_idx]}")
            ax.set_xlabel("% Pixels Modified")
            ax.set_ylabel("Confidence")
            ax.legend(); ax.grid(True)
            results["blackout"] = {
                "image": fig_to_b64(fig),
                "deletionAUC": del_auc,
                "insertionAUC": ins_auc,
            }
        except Exception as e:
            results["blackout"] = {"error": str(e)}

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

def load_model(weights_path: str, model_name: str):
    global MODEL, DEVICE, MODEL_NAME

    # Override config so get_model() picks the right architecture
    import config as cfg
    cfg.MODEL_NAME = model_name
    cfg.NUM_CLASSES = NUM_CLASSES

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = model_name

    from model import get_model
    MODEL = get_model()

    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=DEVICE)

    # Handle both raw state-dicts and training checkpoints
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        MODEL.load_state_dict(checkpoint["model_state_dict"])
    else:
        MODEL.load_state_dict(checkpoint)

    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"Model '{model_name}' loaded on {DEVICE}.")

    # Build a persistent CPU copy for SHAP — doing this once at startup is far
    # cheaper than deepcopy-ing on every SHAP request (Swin weights ~150 MB).
    import copy
    global CPU_MODEL
    CPU_MODEL = copy.deepcopy(MODEL).cpu()
    CPU_MODEL.eval()
    for p in CPU_MODEL.parameters():
        p.requires_grad_(False)
    print("CPU model copy ready for SHAP.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", required=True,
        help="Path to your .pth checkpoint, e.g. checkpoints/efficientnet_best_model.pth"
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name: efficientnet | convnext | swin | raddino | radjepa "
             "(auto-detected from filename if omitted)"
    )
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # Auto-detect model name from filename if not supplied
    model_name = args.model
    if model_name is None:
        fname = os.path.basename(args.weights).lower()
        for candidate in ("efficientnet", "convnext", "swin", "raddino", "radjepa", "resnet50"):
            if candidate in fname:
                model_name = candidate
                break
        if model_name is None:
            model_name = "efficientnet"
            print(f"Warning: could not detect model name from filename, defaulting to '{model_name}'.")

    load_model(args.weights, model_name)
    print(f"\nAPI running at http://localhost:{args.port}")
    print("Endpoints:")
    print("  POST /predict")
    print("  POST /explain/gradcam")
    print("  POST /explain/attention_rollout  (ViT models only)")
    print("  POST /explain/lime")
    print("  POST /explain/shap")
    print("  POST /explain/blackout")
    print("  POST /explain/all\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
