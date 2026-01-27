# GuardianVision 🛡️

**Professional-Grade Automated Visual Compliance Auditor**

GuardianVision monitors real-time video feeds in industrial environments to detect PPE (Personal Protective Equipment) violations (e.g., missing helmets/vests) and unauthorized entry. Built with YOLO11 and FastAPI for the 2026 standard.

## 🚀 Key Features

- **Object Detection**: YOLO11 (Ultralytics) optimized for Apple Silicon (MPS/CoreML).
- **Spatial Logic Engine**: Maps person bounding boxes to PPE objects to flag violations.
- **Multi-Object Tracking**: Uses BoTSORT for temporal consistency.
- **Stateful Alerts**: 10s cooldown mechanism and persistence thresholds (10-frame rule).
- **Automated Auditing**: Captures high-res evidence snapshots of persistent violations.
- **Dashboard**: Next.js 15 App Router with real-time WebSocket streaming and analytics.
- **Edge Acceleration**: Explicit MPS acceleration for M2 Pro/Max Neural Engines.

## 🛠️ Tech Stack

- **Backend**: Python 3.12, FastAPI, Ultralytics YOLO11, OpenCV, Albumentations.
- **Frontend**: Next.js 15 (Typed), Tailwind CSS, Lucide React, WebSockets.
- **Deployment**: CoreML optimization for macOS Edge processing.

## 📦 Installation

### Backend
1. `cd backend`
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python main.py`

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`

## 📊 Performance
- **Inference Speed**: ~20-30 FPS on Apple M2 Pro (MPS).
- **Latency**: <10ms encode-to-stream latency.
- **Accuracy**: Optimized for industrial PPE Kit Detection datasets.

---
*Created by [fozanmujtaba](https://github.com/fozanmujtaba)*
