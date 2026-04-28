# GuardianVision

Real-time visual safety and PPE compliance monitor built with FastAPI, Ultralytics YOLO, OpenCV, and a Next.js dashboard.

## What Works

- FastAPI backend with `/ws`, `/ws/camera`, `/api/analytics`, `/api/violations`, and `/health`.
- Binary WebSocket stream: `[4-byte JSON length][JSON metadata][JPEG frame]`.
- YOLO tracking via ByteTrack for live detections.
- Stateful PPE alerting with a 10-frame persistence threshold and 10-second PPE alert cooldown.
- Evidence snapshots saved under `backend/violations/`.
- Next.js dashboard for live video, alerts, analytics, evidence gallery, and speech synthesis.

## Model Files

Large `.pt` weights are intentionally not committed. At startup the backend tries, in order:

1. `MODEL_PATH` if provided.
2. `models/guardian_vision_v1.pt` from `scripts/train_safety_model.py`.
3. `models/ppe_model.pt` from `scripts/train_ppe.py`.
4. Ultralytics `yolo11n.pt` as a COCO fallback so a fresh clone can boot.

The fallback model detects COCO classes, not PPE. For actual PPE compliance alerts, train or provide one of the safety models above.

## Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Optional:

```bash
python main.py --model ..\models\guardian_vision_v1.pt
set PERSISTENCE_THRESHOLD=10
set FORCE_CPU=true
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Production checks:

```bash
npm run lint
npm run build
npm audit --audit-level=moderate
```

## Training

The portable training scripts live in `scripts/`:

- `prepare_mega_dataset.py` generates `scripts/mega_dataset/` and `scripts/mega_data.yaml`.
- `train_safety_model.py` trains the 24-class safety model and copies the best checkpoint to `models/guardian_vision_v1.pt`.
- `train_ppe.py` trains the 10-class PPE model and copies the best checkpoint to `models/ppe_model.pt`.

Datasets, run outputs, and model weights are ignored by Git. Keep only source code, dataset configs, reports, and reproducibility instructions in the repository.
