"""
Entry point for Render deployment.

Environment variables:
  MODEL_NAME : efficientnet | convnext | swin | raddino | radjepa | resnet50
               (defaults to "efficientnet")
  PORT       : set automatically by Render (default 10000)
"""

import os
import sys
import threading

# ── resolve paths ─────────────────────────────────────────────────────────────
root_path = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(root_path, "src")
for p in (root_path, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── import Flask app (does NOT load model yet) ────────────────────────────────
import api
app = api.app

# ── load model in background thread so port binds immediately ─────────────────
def _load_model_background():
    try:
        model_name = os.environ.get("MODEL_NAME", "efficientnet").strip().lower()
        print(f"[startup] Loading model in background: {model_name}")
        from download_weights import download_weights
        weights_path = download_weights(model_name)
        api.load_model(weights_path, model_name)
        print("[startup] Model ready.")
    except Exception as e:
        print(f"[startup] Model load failed: {e}")
        raise

threading.Thread(target=_load_model_background, daemon=True).start()

# ── local dev fallback ────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
