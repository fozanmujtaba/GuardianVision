import os
import yaml
import shutil
from pathlib import Path

def setup_dataset_structure():
    """
    Creates the directory structure for the merged GuardianVision Mega-Dataset.
    """
    base_path = Path("mega_dataset")
    subdirs = ["images/train", "images/val", "labels/train", "labels/val"]
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created mega_dataset structure at {base_path.absolute()}")

def merge_datasets():
    """Main execution logic to merge all datasets."""
    setup_dataset_structure()
    generate_mega_yaml()
    
    base_dir = Path("/Users/mac/projects/GuardianVision/scripts/datasets/ppe")
    target_dir = Path("mega_dataset")
    
    # Define source paths and their specific remapping
    # Map: {original_id: final_id}
    configs = [
        {
            "name": "PPE",
            "path": base_dir / "data",
            "mapping": {i: i for i in range(10)}, # 0-9 stay same
            "splits": {"train": "images/train", "val": "images/val"}
        },
        {
            "name": "Fire/Smoke",
            "path": base_dir / "fire:smoke detection",
            "mapping": {1: 10, 0: 11}, # Fire->10, Smoke->11
            "splits": {"train": "data/train/images", "val": "data/val/images"}
        },
        {
            "name": "Fall",
            "path": base_dir / "fall_dataset",
            "mapping": {0: 14, 1: 5, 2: 15}, # Fall->14, Walk->5, Sit->15
            "splits": {"train": "images/train", "val": "images/val"}
        },
        {
            "name": "FSE Detection",
            "path": base_dir / "1_FSE Detection",
            "mapping": {0: 16, 1: 17, 2: 18, 3: 13}, # Blanket->16, CallPoint->17, SmokeDet->18, Ext->13
            "splits": {"train": "train/images", "val": "valid/images"}
        },
        {
            "name": "FSE Marking",
            "path": base_dir / "2_FSE Marking Detection",
            "mapping": {0: 19, 1: 20, 2: 21, 3: 12, 4: 22, 5: 23}, # ExitSign->12, etc.
            "splits": {"train": "train/images", "val": "valid/images"}
        }
    ]

    for config in configs:
        print(f"📦 Merging {config['name']}...")
        for split_key, split_path in config['splits'].items():
            img_src = config['path'] / split_path
            
            # Special logic for datasets that don't have separate train/val folders but split by filename
            if not img_src.exists():
                print(f"⚠️ Skip {config['name']} {split_key}: {img_src} not found")
                continue

            # Target split (train or val)
            img_dst = target_dir / "images" / split_key
            lbl_dst = target_dir / "labels" / split_key
            
            for img_file in img_src.glob("*.[jJ][pP][gG]"):
                # Copy Image
                shutil.copy(img_file, img_dst / img_file.name)
                
                # Copy and Remap Label
                lbl_name = img_file.stem + ".txt"
                lbl_src_path = config['path'] / split_path.replace("images", "labels") / lbl_name

                if lbl_src_path.exists():
                    with open(lbl_src_path, "r") as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.split()
                        if not parts: continue
                        old_id = int(parts[0])
                        if old_id in config['mapping']:
                            parts[0] = str(config['mapping'][old_id])
                            new_lines.append(" ".join(parts))
                    
                    with open(lbl_dst / lbl_name, "w") as f:
                        f.write("\n".join(new_lines))

    print("\n✅ MEGA-MERGE COMPLETE!")

def generate_mega_yaml():
    """Generates the YOLO yaml file."""
    data = {
        'path': os.path.abspath("mega_dataset"),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 24,
        'names': [
            "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", 
            "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
            "Fire", "Smoke", "Emergency Exit Sign", "Fire Extinguisher", 
            "Fall Detected", "Sitting", "Fire Blanket", "Manual Call Point", 
            "Smoke Detector", "Wall Hydrant Sign", "Fire Extinguisher Sign Old", 
            "Call Point Sign", "Fire Door Sign", "Fire Extinguisher Sign"
        ]
    }
    with open("mega_data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)

if __name__ == "__main__":
    merge_datasets()
