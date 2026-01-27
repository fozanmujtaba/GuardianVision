# YOLO11 CoreML Export for Apple Silicon (M2 Pro)

To achieve maximum FPS using the Neural Engine, export your YOLO11 model to CoreML format.

## Command Line Interface
Run this in your terminal with the ultralytics environment active:

```bash
# Export to CoreML with NMS support
yolo export model=yolo11n.pt format=coreml nms=True device=mps
```

## Python API Usage
Alternatively, use this snippet in a script:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Export the model
path = model.export(format="coreml", nms=True, device="mps")
print(f"CoreML model saved at: {path}")
```

## Performance Tips
1. **Half Precision**: CoreML exports often benefit from `half=True` to leverage FP16 units.
2. **NMS**: The `nms=True` flag embeds the Non-Maximum Suppression into the CoreML graph, reducing CPU overhead during post-processing.
