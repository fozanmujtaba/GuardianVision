import albumentations as A
import cv2
import os
from glob import glob

def get_industrial_augmentation():
    """
    Simulates industrial conditions:
    1. Motion blur (fast moving machinery)
    2. Low light / Random Brightness (industrial basements)
    3. Gaussian Noise (sensor noise in high temp)
    4. Shadows (RandomShadow)
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3),
        A.MotionBlur(p=0.3, blur_limit=7),
        A.GaussianBlur(p=0.2, blur_limit=5),
        A.RandomShadow(p=0.4, num_shadows_limit=3),
        A.CLAHE(p=0.2), # Improve contrast in low light
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_dataset(input_dir, output_dir):
    aug = get_industrial_augmentation()
    images = glob(os.path.join(input_dir, "*.jpg"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in images:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels (dummy example assuming YOLO format)
        # In practice, you'd load from .txt files
        # [...] 
        
        transformed = aug(image=image, bboxes=[], class_labels=[])
        transformed_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
        
        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, f"aug_{base_name}"), transformed_image)

if __name__ == "__main__":
    print("Industrial Augmentation Pipeline Ready.")
    # Example usage:
    # augment_dataset("data/raw", "data/augmented")
