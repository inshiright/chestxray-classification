import cv2
import torch
import numpy as np
from torchvision import transforms
from model import get_model
from config import IMAGE_SIZE, MODEL_NAME

def run_sentrycam_colab(video_path, model_weights_path, output_path="sentrycam_output.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        top_class_idx = np.argmax(probs)
        top_prob = probs[top_class_idx]
        
        label_text = f"Class {top_class_idx} ({MODEL_NAME}): {top_prob:.2f}"
        
        cv2.putText(
            frame, 
            label_text, 
            (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved successfully to {output_path}")

if __name__ == "__main__":
    # Example Usage:
    # Upload a sample video to Colab before running this
    run_sentrycam_colab("sample_xray_video.mp4", "path_to_saved_model.pth")