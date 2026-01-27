"""
GuardianVision PPE Model Training - HIGH ACCURACY
Trains YOLO11 on the Kaggle PPE Kit Detection dataset.
"""

from ultralytics import YOLO
import torch
import os
import shutil

def train_ppe_model(epochs: int = 100):
    """
    Train YOLO11 on the Kaggle PPE dataset for high accuracy.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Training on device: {device}")
    
    # Path to the manually created data.yaml
    data_yaml = "/Users/mac/projects/GuardianVision/scripts/datasets/ppe/ppe_kaggle.yaml"
    
    if not os.path.exists(data_yaml):
        print(f"❌ Error: {data_yaml} not found!")
        return

    # Load base model
    print(f"📦 Loading base model: yolo11n.pt")
    model = YOLO("yolo11n.pt")
    
    print(f"🏋️ Starting HIGH ACCURACY training for {epochs} epochs...")
    print("⏱️ Estimated time: 1-2 hours on M2 Pro")
    print()
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        device=device,
        project="runs",
        name="ppe_kaggle_prod",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        batch=16,  # Optimized for M2 Pro
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
    )
    
    # Copy best weights to models directory
    best_weights = "runs/ppe_kaggle_prod/weights/best.pt"
    final_path = "../models/ppe_model.pt"
    
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_path)
        print(f"\n✅ Production model saved to: {final_path}")
        print("🔄 Restart the backend to use the new model!")
    else:
        # Check alternate paths
        for root, dirs, files in os.walk("runs/ppe_kaggle_prod"):
            if "best.pt" in files:
                src = os.path.join(root, "best.pt")
                shutil.copy(src, final_path)
                print(f"✅ Found and copied from: {src}")
                break
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("GuardianVision PPE Production Training")
    print("=" * 60)
    print()
    
    train_ppe_model(epochs=100)
