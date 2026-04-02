"""
Entry point for Render deployment.
Replaces the --weights / --model CLI args with environment variables:

  MODEL_NAME   : efficientnet | convnext | swin | raddino | radjepa | resnet50
                 (defaults to "efficientnet")
  PORT         : HTTP port (Render sets this automatically)

Usage (Render start command):
  gunicorn start:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
"""

import os
import sys

# ── resolve paths ─────────────────────────────────────────────────────────────
root_path = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(root_path, "src")
for p in (root_path, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── read env vars ─────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "efficientnet").strip().lower()
PORT = int(os.environ.get("PORT", 5000))

# ── download weights if needed ────────────────────────────────────────────────
from download_weights import download_weights
weights_path = download_weights(MODEL_NAME)

# ── import the Flask app and load the model ───────────────────────────────────
# api.py exposes `app` and `load_model` at module level
import api
api.load_model(weights_path, MODEL_NAME)

# ── expose `app` for gunicorn ─────────────────────────────────────────────────
app = api.app

# ── local dev fallback ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nAPI running at http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
