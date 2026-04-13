"""
Downloads model weights from HuggingFace if not already cached locally.
Called automatically by start.py on startup.
"""

import os
import urllib.request

HF_BASE = "https://huggingface.co/lyj9900/RAD14NT/resolve/main/models"

MODEL_FILES = {
    "efficientnet": "efficientnet_best_model.pth",
    "convnext":     "convnext_best_model.pth",
    "swin":         "swin_best_model.pth",
    "raddino":      "raddino_best_model.pth",
    "radjepa":      "radjepa_best_model.pth",
    "resnet50":     "resnet50_best_model.pth",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def download_weights(model_name: str) -> str:
    """
    Returns the local path to the .pth file for the given model name.
    Downloads from HuggingFace if not already present in ./checkpoints/.
    """
    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_FILES.keys())}"
        )

    filename = MODEL_FILES[model_name]
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_path = os.path.join(CACHE_DIR, filename)

    if os.path.exists(local_path):
        print(f"[weights] Found cached: {local_path}")
        return local_path

    url = f"{HF_BASE}/{filename}?download=true"
    print(f"[weights] Downloading {filename} from HuggingFace...")
    print(f"[weights] URL: {url}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r[weights] {pct:.1f}%  ({downloaded // 1_000_000} MB / {total_size // 1_000_000} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, local_path, reporthook=_progress)
    print(f"\n[weights] Saved to: {local_path}")
    return local_path


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "efficientnet"
    path = download_weights(name)
    print(f"Ready: {path}")
