import torch
from ultralytics import YOLO
import os

def train_safety_model():
    print("🚀 Initializing Safety Model Training (24 Classes)")
    
    # Check for MPS (Apple Silicon Acceleration)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"💻 Using Device: {device}")
    
    # PATHS
    base_proj = os.path.abspath("runs/detect/runs/models")
    last_weights = os.path.join(base_proj, "guardian_vision_v1/weights/last.pt")
    data_yaml = os.path.abspath("mega_data.yaml")
    
    # Initialize model: Load last checkpoint if it exists, otherwise start fresh
    if os.path.exists(last_weights):
        print(f"🔄 Resuming from last checkpoint: {last_weights}")
        # IMPORTANT: To resume properly, load the path string directly into YOLO()
        model = YOLO(last_weights)
        resume = True
    else:
        print("🆕 Starting fresh training from yolo11n.pt")
        model = YOLO("yolo11n.pt")
        resume = False
    
    # Run Training
    # If resuming, Ultralytics recommends minimal args, but we keep batch for stability
    model.train(
        data=data_yaml,
        epochs=15,           # Capped at 15 for safety/heat management
        imgsz=640,
        batch=8,             
        device=device,
        resume=resume,       
        name="guardian_vision_v1",
        project=base_proj,
        exist_ok=True,
        verbose=True
    )
    
    print("✅ Training sequence initiated. Results will be saved to runs/models/guardian_vision_v1")

if __name__ == "__main__":
    train_safety_model()
