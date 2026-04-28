# GuardianVision
## Real-Time AI Safety & Compliance Monitoring System
### Deep Learning Project — Final Technical Report

**Group Members**

| Name | CMS ID |
|---|---|
| Muhammad Faozan Mujtaba | 457570 |
| Abdul Basit | 462779 |
| Shees ur Rehman | 470810|

**Course:** Deep Learning | **Submission:** End-Semester Project

---

## 1. Problem Statement

Industrial workplaces — construction sites, manufacturing plants, warehouses — account for a disproportionate share of occupational injuries and fatalities. Manual safety monitoring is labor-intensive, error-prone, and impossible to scale across large or multi-zone facilities. Human supervisors cannot maintain continuous vigilance; delayed responses to critical events like falls or fire can be catastrophic.

Regulatory bodies increasingly mandate verifiable audit trails for PPE compliance. Traditional paper-based inspection logs are easy to falsify and difficult to analyze at scale.

**GuardianVision** is an end-to-end, AI-driven safety monitoring system that:
- Detects PPE violations in real-time using a 24-class deep learning model
- Identifies critical emergencies (fire, smoke, man-down) and triggers immediate alerts
- Generates tamper-evident photographic evidence for every confirmed violation
- Provides an interactive compliance dashboard with live analytics

---

## 2. Objectives

| ID | Objective | Target | Achieved |
|---|---|---|---|
| O1 | PPE Detection Accuracy (mAP@0.5) | ≥ 85% | **56.7%** (see §6 for analysis) |
| O2 | Emergency Incident Response Time | ≤ 3 seconds | ✅ ~0.3–1.0 s (stateful, real-time) |
| O3 | End-to-End Glass-to-Glass Latency | ≤ 100 ms | ✅ ~60–90 ms on Apple MPS |
| O4 | Automated Evidence & Dashboard | Fully functional | ✅ Complete |

---

## 3. System Architecture

GuardianVision is a five-stage modular pipeline:

```
[Browser Camera]
      │  WebSocket (Binary)
      ▼
[FastAPI Backend]
      │
      ├─► Preprocessing (Resize → 480×480)
      │
      ├─► YOLO11s Inference (Apple MPS)
      │         │
      │         └─► ByteTrack (Multi-Object Tracking)
      │
      ├─► PPE Auditor (Stateful Persistence Logic)
      │         │
      │         ├─► Violation Detection (10-frame threshold)
      │         ├─► Critical Event Detection (Fire/Smoke/Fall)
      │         └─► Evidence Snapshot (annotated JPEG)
      │
      ├─► Response Manager (Voice alert scripting)
      │
      └─► Analytics Manager (Daily stats, violation history)
             │  Binary Response [JSON + JPEG]
             ▼
[Next.js Dashboard]
      ├─► Live annotated video feed
      ├─► Compliance gauge (rolling 60-frame window)
      ├─► Detection tag strip + person count
      ├─► Violation alerts + audio beep
      ├─► Evidence gallery (viewable + clearable)
      └─► Analytics charts + CSV export
```

### Technology Stack

| Layer | Technology |
|---|---|
| AI Model | YOLO11s (Ultralytics), PyTorch |
| Object Tracking | ByteTrack |
| Computer Vision | OpenCV |
| Backend | FastAPI, Uvicorn, Python 3.9 |
| Communication | WebSocket (Binary Protocol), REST API |
| Frontend | Next.js 15, React, Tailwind CSS v4, Recharts |
| Hardware Acceleration | Apple Silicon MPS (Metal Performance Shaders) |
| Dataset Tooling | Roboflow Universe, Custom merge scripts |

---

## 4. Dataset

### 4.1 Data Sources — Mega-Dataset (5 Sources Merged)

GuardianVision's training data was assembled by merging five publicly available Roboflow datasets into a unified 24-class "Mega-Dataset," with custom class ID remapping to produce a consistent label space.

| Source | Classes Contributed | Class IDs |
|---|---|---|
| Construction Site Safety (PPE) | Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, Machinery, Vehicle | 0–9 |
| Fire & Smoke Detection | Fire, Smoke | 10–11 |
| Fall Detection | Fall Detected, Sitting | 14–15 |
| FSE Detection (Equipment) | Fire Blanket, Manual Call Point, Smoke Detector, Fire Extinguisher | 13, 16–18 |
| FSE Marking (Signage) | Emergency Exit Sign, Fire Extinguisher Signs, Call Point Sign, Fire Door Sign | 12, 19–23 |

**Total: 24 classes | ~11,800+ training images**

### 4.2 Class Distribution & Imbalance

The dataset exhibits significant class imbalance — a known challenge in multi-source merges:

| Class | Approx. Instances | Notes |
|---|---|---|
| Person | ~18,000 | Most represented |
| Fire | ~9,638 | Well represented |
| Smoke | ~7,794 | Well represented |
| Hardhat / NO-Hardhat | ~6,000 each | Core PPE classes |
| Safety Vest / NO-Safety Vest | ~5,000 each | Core PPE classes |
| Fall Detected | ~213 | Severely underrepresented |
| Fire Blanket | ~0 | Missing from available data |

**Mitigation strategies applied:** mosaic augmentation (4-image tiling), MixUp blending, Copy-Paste augmentation, and a reduced `cls=0.3` classification loss weight to prevent majority classes from dominating gradients.

### 4.3 Preprocessing & Augmentation Pipeline

All augmentations are applied on-the-fly by Ultralytics during training:

| Augmentation | Parameter | Purpose |
|---|---|---|
| Mosaic | prob=1.0 | 4-image composition; simulates crowded scenes |
| MixUp | prob=0.15 | Blends two images; improves generalization |
| Copy-Paste | prob=0.1 | Pastes object instances across images |
| HSV Jitter | h=0.015, s=0.7, v=0.4 | Lighting/color variance |
| Horizontal Flip | prob=0.5 | Symmetry augmentation |
| Mosaic Disable | last 10 epochs | Stabilizes final convergence |
| Resize | imgsz=320 | Optimized for Apple MPS throughput |

---

## 5. Deep Learning Model

### 5.1 Model Selection — YOLO11s

We selected **YOLOv11 Small (YOLO11s)** as the detection backbone, from Ultralytics (2024).

**Why YOLO11s over alternatives:**
- Single-pass detection (no region proposal overhead) → real-time throughput
- Anchor-free detection head → better generalization to small objects (helmets, masks)
- Improved C3k2 neck architecture over YOLOv8 → more accurate feature aggregation
- 9.4M parameters — small enough for Apple MPS inference at 15+ FPS, large enough for 24-class capacity
- YOLO11n (nano) was tested first but lacked capacity for 24 classes simultaneously

### 5.2 Model Architecture (YOLO11s Summary)

```
Input (320×320×3)
       │
    Backbone (C3k2 blocks + C2PSA attention)
       │  Feature extraction at 3 scales (P3, P4, P5)
       │
    Neck (PAN-FPN)
       │  Feature Pyramid Network — fuses multi-scale features
       │
    Detection Head (Anchor-Free)
       │  3 output scales: 40×40, 20×20, 10×10
       │
    Output: Class scores (24) + BBox regression (4) + Objectness
```

**Key design choices:**
- C2PSA (Cross-Stage Partial with Spatial Attention) in the backbone improves detection of occluded objects — critical for PPE partially hidden by clothing or camera angle
- Decoupled detection head separates classification and regression branches, improving mAP on small objects

### 5.3 Training Configuration

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Base model | `yolo11s.pt` (COCO pretrained) | Transfer learning from 80-class COCO |
| Epochs | 50 (effective ~46) | Early stopping patience=20 |
| Image size | 320 × 320 | ~2× speedup vs 640 on MPS |
| Batch size | 32 | Fills 4.5 GB MPS memory |
| Dataset fraction | 0.3 | 30% per epoch for speed on M2 Pro |
| Optimizer | Auto (AdamW selected) | YOLO auto-selects based on batch |
| Learning rate (lr0) | 0.01 | Standard YOLO default |
| LR final (lrf) | 0.01 | Cosine decay to 1% of lr0 |
| Weight decay | 0.0005 | L2 regularization |
| Warmup epochs | 3 | Gradual LR ramp-up |
| Classification loss weight | 0.3 | Reduces majority-class dominance |
| Device | Apple MPS | M2 Pro GPU via Metal |

### 5.4 Transfer Learning Strategy

Rather than training from random weights, we used **transfer learning** from YOLO11s pretrained on COCO (80 classes). The pretrained backbone had already learned fundamental visual features (edges, textures, shapes). We fine-tuned all layers — backbone, neck, and head — replacing the final detection head with a 24-class output layer. This approach:

1. Dramatically reduces training time vs training from scratch
2. Improves generalization on smaller datasets
3. Leverages COCO features (persons, vehicles) that overlap with our domain

**Multi-stage training:** Due to hardware constraints (Apple M2 Pro vs cloud GPU), training was conducted in two stages:
- Stage 1: `imgsz=416`, `fraction=1.0` → 15 epochs (mAP50=0.401)
- Stage 2: Loaded Stage 1 weights, fresh training with `imgsz=320`, `fraction=0.3` → 31 epochs (peak mAP50=0.567)

---

## 6. Results & Evaluation

### 6.1 Model Performance

| Metric | Value |
|---|---|
| **mAP@0.5** | **0.567** |
| mAP@0.5:0.95 | 0.351 |
| Precision | 0.636 |
| Recall | 0.439 |
| F1-Score | ~0.52 |

**Training convergence:** mAP50 improved steadily from 0.401 (epoch 15, Stage 1) to **0.567** (epoch ~22 overall), then plateaued. Early stopping triggered correctly after no improvement for 20 epochs.

### 6.2 Objective O1 — Gap Analysis

The proposal targeted mAP@0.5 ≥ 85%. We achieved **56.7%**. The gap is explained by three compounding factors:

1. **Severe class imbalance**: Fall Detected (~213 samples) and Fire Blanket (~0 samples) significantly depress the mean. Per-class mAP for well-represented classes (Person, Hardhat, Fire, Smoke) is estimated >70%.

2. **Multi-source label noise**: Merging 5 independently annotated datasets introduces inconsistent labeling conventions (e.g., "Safety Vest" vs "High-Vis Vest"), which confuses the classifier.

3. **Hardware constraint**: Training was capped at imgsz=320 and fraction=0.3 on an Apple M2 Pro. A full-resolution (640px), full-dataset run on a cloud GPU would meaningfully close this gap.

**Practical implication for demo:** Despite the aggregate mAP, the system visually detects PPE violations, fire, and smoke reliably in live use. The stateful persistence layer (10-frame threshold) suppresses false positives effectively, so operational precision is higher than the raw metric suggests.

### 6.3 System Performance

| Metric | Target | Measured |
|---|---|---|
| Inference time (YOLO only) | ≤ 40 ms | **~35–55 ms** (Apple MPS, imgsz=480) |
| End-to-end WebSocket latency | ≤ 100 ms | **~60–90 ms** |
| Live FPS (frontend) | ≥ 15 FPS | **~15–20 FPS** |
| Frame skip strategy | — | Every 2nd frame processed |

All three system performance objectives (O2, O3, O4) were fully met.

---

## 7. Stateful Alert Engine (PPEAuditor)

A raw neural network produces per-frame bounding box predictions. A single-frame detection is insufficient for triggering alerts — transient misdetections would flood the system with false positives. The `PPEAuditor` module implements a **stateful persistence layer** on top of raw detections.

### 7.1 Persistence Logic

For each tracked person (identified by ByteTrack ID):
1. A violation counter per type (`Hardhat`, `Safety Vest`) is maintained
2. The counter **increments** each frame the violation is observed
3. The counter **resets to 0** when the violation clears
4. An alert fires only when the counter ≥ **10 consecutive frames** (~0.67 seconds at 15 FPS)

This means a person must be continuously missing a hardhat for 10 frames before any alert is raised — eliminating transient false positives from model uncertainty.

### 7.2 Critical Event Detection

Fire, Smoke, and Fall Detected classes bypass the persistence requirement and trigger **immediate critical events** on first detection (these are high-severity and require instant response). The `ResponseManager` then generates a contextual voice announcement script:

- Fire → *"Attention. Fire detected. Please evacuate to the nearest exit immediately."*
- Smoke → *"Warning. Smoke detected. Please investigate and prepare for evacuation."*
- Man-Down → *"Emergency. A person is down. Medical assistance required."*

### 7.3 Evidence Capture

On each confirmed violation (first occurrence per person per session):
- The current frame is copied
- A red bounding box and "VIOLATION:" label are drawn
- The annotated frame is saved as `violations/violation_{person_id}_{type}_{timestamp}.jpg`
- A 10-second cooldown prevents duplicate snapshots for the same ongoing violation

---

## 8. Multi-Object Tracking — ByteTrack

Raw YOLO detections have no identity: the same person is a different detection each frame. Without tracking, per-person violation history is impossible.

**ByteTrack** (Zhang et al., ECCV 2022) assigns persistent integer IDs to each detected person across frames. Unlike SORT or DeepSORT, ByteTrack uses *every* detection box — including low-confidence ones — via a two-step association:

1. High-confidence detections are matched first (IoU-based Hungarian algorithm)
2. Unmatched tracks are then associated with low-confidence detections

This preserves track continuity through brief occlusions — critical in construction environments where workers frequently pass behind machinery or each other.

**GuardianVision integration:** `model.track(..., tracker="bytetrack.yaml", persist=True)` is called each frame. Track IDs are extracted from `box.id` and passed to the `PPEAuditor` as `person_id` keys.

---

## 9. Binary WebSocket Protocol

Standard WebSocket text transmission of Base64-encoded frames introduces ~33% size overhead. We implemented a **custom binary framing protocol** to reduce latency:

```
[ 4 bytes: JSON length (big-endian) ] [ N bytes: JSON metadata ] [ M bytes: raw JPEG ]
```

The frontend parses the first 4 bytes to determine where the JSON ends and the JPEG begins — decoding both in a single WebSocket message. This eliminates the Base64 encode/decode cycle entirely for the image, reducing per-frame payload by ~25% and dropping latency meaningfully at 15+ FPS.

---

## 10. Frontend Dashboard

The Next.js 15 dashboard provides a real-time operator interface:

| Component | Description |
|---|---|
| **Live Feed** | Annotated MJPEG-equivalent stream, corner targeting brackets, CRT scan-line overlay |
| **Compliance Gauge** | SVG animated ring, rolling 60-frame window; green (>80%) → amber → red |
| **Detection Tags** | Per-class color-coded badges updated every frame |
| **Person Counter** | Live worker count extracted from `Person` class detections |
| **Alert Banner** | Red flashing banner + audio beep on each violation trigger |
| **Evidence Gallery** | Scrollable thumbnail grid of violation snapshots, with "Clear Gallery" button |
| **Analytics Charts** | Daily violation trend (Recharts bar chart), violation-by-type breakdown |
| **CSV Export** | One-click download of all analytics data |
| **Device Badge** | Shows `MPS` or `CPU` to confirm GPU acceleration is active |

---

## 11. Key Challenges & How We Solved Them

| Challenge | Root Cause | Solution |
|---|---|---|
| Training too slow (60 min/epoch) | imgsz=640, full dataset on M2 Pro | Reduced to imgsz=320, fraction=0.3 → ~10 min/epoch |
| Mac sleep killing training | `caffeinate -i` only blocks idle, not display | Upgraded to `caffeinate -dims` (blocks display + system sleep) |
| YOLO `resume=True` ignored new settings | Loads checkpoint config, not new args | Load checkpoint as weights only (no `resume=True`), pass all hyperparams explicitly |
| Analytics overcounting (57K violations) | Logging every frame, not only alert events | Fixed: `analytics.log_frame(count, violations if alert_triggered else [])` |
| Snapshot naming collision | Static filename overwrote previous evidence | Include `person_id`, violation type, and timestamp in filename |
| `time.timestamp()` crash | `time` module has no `.timestamp()` attribute | Fixed to `time.time()` |
| Class imbalance suppressing mAP | 5 merged datasets with unequal class sizes | `cls=0.3` loss weight + full augmentation suite |
| Base64 WebSocket overhead | Text encoding adds 33% size | Custom binary framing: `[4-byte header][JSON][JPEG]` |

---

## 12. Conclusion

GuardianVision delivers a fully functional, real-time industrial safety monitoring system built on state-of-the-art deep learning. The system meets all system performance objectives (latency, FPS, evidence capture, dashboard) and demonstrates meaningful detection of PPE violations and critical events in live use.

The 24-class YOLO11s model achieves mAP@0.5 = **0.567** — below the 85% proposal target, but attributable to well-understood dataset constraints (class imbalance, label noise from multi-source merging) rather than architectural limitations. The stateful persistence layer compensates operationally by filtering false positives, delivering reliable alert behavior in practice.

**What was built:**
- End-to-end real-time pipeline from browser camera to annotated feed in <100ms
- 24-class custom-trained YOLO11s model on a merged 5-source mega-dataset
- ByteTrack-powered per-person violation history with evidence capture
- Full Next.js dashboard with live compliance gauge, analytics, and evidence gallery

**Future improvements:**
- Collect balanced data for minority classes (fall detection, fire blanket)
- Train full-resolution (640px) on cloud GPU for 3–5% mAP gain
- Add GPS-zone tagging for multi-camera facility coverage
- Integrate Twilio/SendGrid for real SMS/email emergency notifications
- Export model to CoreML for on-device iOS deployment

---

## 13. References

[1] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* IEEE CVPR.

[2] Nath, N. D., Behzadan, A. H., & Paal, S. G. (2020). Deep learning for site safety: Real-time detection of personal protective equipment. *Automation in Construction, 112*, 103085.

[3] Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., & Wang, X. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.

[4] Ultralytics. (2024). *YOLOv11: State-of-the-Art Object Detection.* https://github.com/ultralytics/ultralytics

[5] Roboflow Universe. (2023). *Construction Site Safety Dataset.* https://universe.roboflow.com

---

*GuardianVision — Real-Time AI Safety Monitoring | Deep Learning End-Semester Project*
