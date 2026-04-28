"""
GuardianVision 24-Class Safety Model Training — v2
Key fixes over v1:
  - yolo11s (small) instead of nano — more capacity for 24 classes
  - 50 epochs at imgsz=416, batch=32 — ~20h on M2 Pro
  - Full augmentation suite (mosaic, mixup, copy_paste)
  - close_mosaic=10 for stable final convergence
  - patience=20 early stopping
  - Auto-resumes from last.pt if interrupted
  - Copies best.pt to models/ on completion
"""

import torch
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNS_DIR = SCRIPT_DIR / "runs"

DATA_YAML = SCRIPT_DIR / "mega_data.yaml"
OUTPUT_NAME = "guardian_v2"
LAST_CKPT = RUNS_DIR / OUTPUT_NAME / "weights" / "last.pt"
FINAL_PATH = PROJECT_ROOT / "models" / "guardian_vision_v1.pt"


def train(epochs: int = 50):
    if not DATA_YAML.exists():
        print(f"❌  mega_data.yaml not found at {DATA_YAML}")
        print("    Run prepare_mega_dataset.py first.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load checkpoint weights (without resume=True so new hyperparams take effect)
    if LAST_CKPT.exists():
        print(f"🔄  Continuing from checkpoint weights: {LAST_CKPT}")
        print(f"🚀  Device: {device}")
        model = YOLO(str(LAST_CKPT))
        model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=320,
            device=device,
            batch=32,
            fraction=0.3,

            project=str(RUNS_DIR),
            name=OUTPUT_NAME,
            exist_ok=True,
            save=True,
            plots=True,
            verbose=True,

            patience=20,

            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,

            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.1,
            close_mosaic=10,

            cls=0.3,
        )
    else:
        print(f"🆕  Starting fresh training")
        print(f"🚀  Device        : {device}")
        print(f"📦  Base model    : yolo11s.pt  (9.4M params)")
        print(f"🗂   Dataset       : {DATA_YAML}")
        print(f"🔁  Epochs        : {epochs}  (early stop patience=20)")
        print(f"⏱   Estimated time: ~20 h on M2 Pro  (imgsz=416, batch=32)")
        print()

        model = YOLO("yolo11s.pt")
        model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=320,
            device=device,
            batch=32,
            fraction=0.3,

            project=str(RUNS_DIR),
            name=OUTPUT_NAME,
            exist_ok=True,
            save=True,
            plots=True,
            verbose=True,

            patience=20,

            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,

            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.1,
            close_mosaic=10,

            cls=0.3,
        )

    # Copy best weights to models directory
    best = RUNS_DIR / OUTPUT_NAME / "weights" / "best.pt"
    if not best.exists():
        for root, _, files in os.walk(RUNS_DIR / OUTPUT_NAME):
            if "best.pt" in files:
                best = Path(root) / "best.pt"
                break

    if best.exists():
        FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, FINAL_PATH)
        print(f"\n✅  Saved to: {FINAL_PATH}")
        print("🔄  Restart the backend — it will auto-load the 24-class model.")
    else:
        print(f"\n⚠️  Could not find best.pt — check {RUNS_DIR / OUTPUT_NAME / 'weights'}")


if __name__ == "__main__":
    print("=" * 60)
    print("  GuardianVision — 24-Class Safety Model Training v2")
    print("=" * 60)
    print()
    train(epochs=35)
