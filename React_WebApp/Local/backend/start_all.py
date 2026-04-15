import subprocess
import sys
import os
import signal
import time

# List of models to launch
MODELS = [
    {"name": "resnet50", "port": 5001, "weights": "checkpoints/resnet50_best_model.pth"},
    {"name": "efficientnet", "port": 5002, "weights": "checkpoints/efficientnet_best_model.pth"},
    {"name": "convnext", "port": 5003, "weights": "checkpoints/convnext_best_model.pth"},
    {"name": "swin", "port": 5004, "weights": "checkpoints/swin_best_model.pth"},
    {"name": "raddino", "port": 5005, "weights": "checkpoints/raddino_best_model.pth"},
    {"name": "radjepa", "port": 5006, "weights": "checkpoints/radjepa_best_model.pth"},
]

processes = []

def signal_handler(sig, frame):
    """Gracefully terminate all child processes on Ctrl+C."""
    print("\n[orchestrator] Stopping all models...")
    for p in processes:
        p.terminate()
    print("[orchestrator] All processes terminated. Done.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    python_exe = sys.executable
    print(f"[orchestrator] Using Python interpreter: {python_exe}")
    print(f"[orchestrator] Working directory: {backend_dir}")
    
    for model in MODELS:
        weights_path = os.path.join(backend_dir, model["weights"])
        if os.path.exists(weights_path):
            print(f"[orchestrator] Launching {model['name']} on port {model['port']}...")
            
            # Use 'python api.py ...' command
            cmd = [
                python_exe, "api.py",
                "--weights", model["weights"],
                "--model", model["name"],
                "--port", str(model["port"])
            ]
            
            # On Windows, we launch in new console windows to keep logs separate and visible
            # This mimics the behavior of the .bat file without the path issues.
            p = subprocess.Popen(
                cmd, 
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            processes.append(p)
        else:
            print(f"[orchestrator] Skipping {model['name']} — Weight file not found: {model['weights']}")
    
    if not processes:
        print("[orchestrator] ERROR: No models were launched. Please ensure weights are in the 'checkpoints' folder.")
        return

    print("\n" + "="*70)
    print("  SUCCESS: ALL AVAILABLE MODEL SERVERS LAUNCHED")
    print("  Keep this window open. Press Ctrl+C here to shut down all servers.")
    print("="*70 + "\n")
    
    # Keep the main process alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
