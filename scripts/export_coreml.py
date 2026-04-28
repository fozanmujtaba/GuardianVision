"""
GuardianVision Model Optimization - PRO GRADE
Exports the trained PPE model to CoreML (FP16) for Mac Neural Engine acceleration.
"""

from ultralytics import YOLO
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "guardian_vision_v1.pt"


def export_to_coreml(model_path: str = str(DEFAULT_MODEL)):
    """
    Export YOLO model to CoreML format.
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found!")
        return

    print(f"📦 Loading model for export: {model_path}")
    model = YOLO(model_path)
    
    print("🚀 Exporting to CoreML (FP16)...")
    print("💡 This will allow the model to run on the Apple Neural Engine (ANE).")
    
    # Export to CoreML
    # nms=True adds Non-MaxPooling Suppression to the model itself
    export_path = model.export(format="coreml", nms=True)
    
    print()
    print("✅ Export Complete!")
    print(f"📄 CoreML Model: {export_path}")
    print("🛠️ To use this in production, update MODEL_PATH in main.py to point to the .mlpackage")

if __name__ == "__main__":
    export_to_coreml()
