"""
GuardianVision Backend - FastAPI with YOLO11 PPE Detection
Supports both WebSocket frame reception and local webcam capture.
"""

import cv2
import base64
import json
import torch
import asyncio
import argparse
import os
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from ultralytics import YOLO
from auditor import DEFAULT_SAFETY_CLASSES, PPEAuditor
from camera import CameraStream, FrameSimulator
from analytics import AnalyticsManager
from response_manager import response_manager

# Configuration (Defaults)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
VIOLATIONS_DIR = BASE_DIR / "violations"
ANALYTICS_PATH = BASE_DIR / "analytics_stats.json"

DEFAULT_MODEL_PATH = MODELS_DIR / "guardian_vision_v1.pt"
LEGACY_MODEL_PATH = MODELS_DIR / "ppe_model.pt"
FALLBACK_MODEL = "yolo11n.pt"
VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Global state
model = None
device = None
auditor = None
camera_task = None
analytics = None
loaded_model_ref = None
SIMULATION_MODE = os.environ.get("SIMULATION_MODE", "false").lower() == "true"


def _normalize_model_ref(model_ref: str) -> str:
    """Resolve local model paths relative to the project when possible."""
    raw_path = Path(model_ref).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)

    for base in (Path.cwd(), PROJECT_ROOT, BASE_DIR):
        candidate = (base / raw_path).resolve()
        if candidate.exists():
            return str(candidate)

    return model_ref


def _coerce_model_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def _detection_name(det):
    return str(det.get("class_name", "")).strip().lower()


def _is_person_detection(det):
    return _detection_name(det) == "person"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    global model, device, auditor, analytics, loaded_model_ref
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if os.environ.get("FORCE_CPU", "false").lower() == "true":
        device = "cpu"
    print(f"🚀 Device: {device}")
    
    # Determine model path dynamically
    current_model_path = _normalize_model_ref(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not os.path.exists(current_model_path) and os.path.exists(LEGACY_MODEL_PATH):
        current_model_path = str(LEGACY_MODEL_PATH)
    
    # Load custom PPE model if available, otherwise fallback
    if os.path.exists(current_model_path):
        print(f"📦 Loading custom safety model: {current_model_path}")
        model = YOLO(current_model_path)
        loaded_model_ref = current_model_path
    else:
        print(f"⚠️ Custom model not found at {current_model_path}, using fallback: {FALLBACK_MODEL}")
        model = YOLO(FALLBACK_MODEL)
        loaded_model_ref = FALLBACK_MODEL
    
    model.to(device)
    # Sync CLASS_NAMES to the loaded model's labels
    CLASS_NAMES.clear()
    CLASS_NAMES.update(_coerce_model_names(model.names))
    print(f"📋 Classes: {list(CLASS_NAMES.values())}")
    persistence_threshold = int(os.environ.get("PERSISTENCE_THRESHOLD", "10"))
    auditor = PPEAuditor(
        cooldown_seconds=10,
        persistence_threshold=persistence_threshold,
        class_names=CLASS_NAMES,
        snapshot_dir=VIOLATIONS_DIR,
    )
    analytics = AnalyticsManager(stats_file=str(ANALYTICS_PATH))
    
    # Background task to save analytics every 60 seconds
    async def save_loop():
        while True:
            await asyncio.sleep(60)
            analytics.save_stats()
            
    asyncio.create_task(save_loop())
    
    print("✅ GuardianVision Backend Ready")
    yield
    
    print("👋 Shutting down...")

app = FastAPI(title="GuardianVision Backend", lifespan=lifespan)

# Static files for violation snapshots
app.mount("/violations", StaticFiles(directory=str(VIOLATIONS_DIR)), name="violations")

@app.get("/")
async def root():
    """Root endpoint providing basic API info."""
    return {
        "name": "GuardianVision API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "analytics": "/api/analytics",
            "violations": "/api/violations",
            "ws": "/ws",
            "ws_camera": "/ws/camera"
        }
    }

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Safety classes are replaced at startup with the loaded model's labels.
CLASS_NAMES = DEFAULT_SAFETY_CLASSES.copy()

def preprocess_frame(frame):
    """OpenCV processing: resize to 480x480 for faster inference."""
    frame = cv2.resize(frame, (480, 480))
    return frame

def process_detections(results):
    """Extract detections from YOLO results with tracking IDs."""
    detections = []
    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            # Get track ID if available, otherwise use index
            track_id = int(box.id[0]) if box.id is not None else i
            detections.append({
                "id": track_id,
                "bbox": box.xyxy[0].tolist(),
                "conf": float(box.conf[0]),
                "class": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            })
    return detections

def annotate_frame(frame, detections, violations):
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        # Default color: Gray
        color = (150, 150, 150)
        class_name = _detection_name(det)
        
        if class_name == "person":
            color = (255, 200, 0)  # Cyan for person
        elif class_name in {"fire", "smoke", "fall detected", "fall"}:
            color = (0, 0, 255) # Red for critical
        elif class_name in {"hardhat", "helmet", "safety vest", "vest"}:
            color = (0, 255, 255) # Yellow/Gold
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['conf']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Highlight violations in red
    for v in violations:
        x1, y1, x2, y2 = map(int, v["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        missing = ", ".join(v["violations"])
        cv2.putText(frame, f"MISSING: {missing}", (x1, y1 - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for frame processing."""
    await websocket.accept()
    print("🔌 Client connected")
    
    import time
    frame_count = 0
    try:
        while True:
            # Handle both Text (Base64) and Binary (Blob)
            msg = await websocket.receive()
            frame_count += 1
            
            if msg["type"] == "websocket.disconnect":
                print("🔌 Client disconnected (graceful)")
                break
                
            if "text" in msg:
                data = msg["text"]
                # Decode base64 image
                try:
                    if "," in data:
                        _, encoded = data.split(",", 1)
                    else:
                        encoded = data
                    img_data = base64.b64decode(encoded)
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"Frame decode error: {e}")
                    continue
            elif "bytes" in msg:
                data = msg["bytes"]
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                continue

            if frame_count % 30 == 0:
                print(f"📦 Received frame ${frame_count}")
            
            if frame is None:
                continue

            processed_frame = preprocess_frame(frame)
            
            # Run inference with tracking (ByteTrack) - Faster than BoTSORT
            inf_start = time.time()
            results = model.track(processed_frame, device=device, persist=True, verbose=False, imgsz=480, tracker="bytetrack.yaml", conf=0.15)
            inf_time = (time.time() - inf_start) * 1000

            detections = process_detections(results)
            if frame_count % 30 == 0:
                print(f"⏱️ Frame {frame_count}: {inf_time:.2f}ms | detections={len(detections)} | classes={[d['class_name'] for d in detections]}")
            
            # Audit for violations and emergency threats
            violations, alert_triggered, critical_events = auditor.audit_frame(detections, frame=processed_frame)
            
            # Simulation Mode: Inject synthetic events if enabled
            if SIMULATION_MODE:
                import time
                # Every 30 seconds, inject a random simulation event
                if int(time.time()) % 60 < 2:
                    critical_events.append({"type": "SIMULATED_FIRE", "location": "Backend Server Room", "bbox": [100, 100, 300, 300]})
                elif 30 <= int(time.time()) % 60 < 32:
                    critical_events.append({"type": "SIMULATED_MAN_DOWN", "location": "Construction Zone A", "bbox": [200, 200, 400, 400]})
            
            # Response Manager (Critical alerts)
            response_actions = response_manager.process_critical_events(critical_events)
            
            # Log analytics — only count violation events, not every frame
            person_count = sum(1 for d in detections if _is_person_detection(d))
            analytics.log_frame(person_count, violations if alert_triggered else [])

            # Annotate frame
            annotated = annotate_frame(processed_frame.copy(), detections, violations)
            
            # Binary Response: Reduce Base64 overhead
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
            
            response_metadata = {
                "detections": detections,
                "violations": violations,
                "critical_events": critical_events,
                "alert": alert_triggered,
                "response_actions": response_actions,
                "device": device
            }
            
            metadata_json = json.dumps(response_metadata).encode('utf-8')
            # Format: [4 bytes JSON length] [JSON] [JPEG]
            message = len(metadata_json).to_bytes(4, byteorder='big') + metadata_json + buffer.tobytes()
            
            await websocket.send_bytes(message)
            
    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": "Internal error", "details": str(e)}))
        except:
            pass

@app.websocket("/ws/camera")
async def camera_stream_endpoint(websocket: WebSocket):
    """WebSocket endpoint that streams from local webcam."""
    await websocket.accept()
    print("📹 Camera stream started")
    
    camera = CameraStream(source=0, fps=15)
    
    try:
        async for frame_b64 in camera.stream_frames():
            # Decode frame for processing
            encoded = frame_b64.split(",")[1]
            img_data = base64.b64decode(encoded)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            processed_frame = preprocess_frame(frame)
            # Run inference with tracking (BoTSORT)
            results = model.track(processed_frame, device=device, persist=True, verbose=False)
            detections = process_detections(results)
            # Audit for violations and emergency threats
            violations, alert_triggered, critical_events = auditor.audit_frame(detections, frame=processed_frame)
            
            # Simulation Mode: Inject synthetic events if enabled
            if SIMULATION_MODE:
                import time
                if int(time.time()) % 60 < 2:
                    critical_events.append({"type": "SIMULATED_FIRE", "location": "Backend Server Room", "bbox": [100, 100, 300, 300]})
                elif 30 <= int(time.time()) % 60 < 32:
                    critical_events.append({"type": "SIMULATED_MAN_DOWN", "location": "Construction Zone A", "bbox": [200, 200, 400, 400]})

            # Response Manager
            response_actions = response_manager.process_critical_events(critical_events)
            
            # Log analytics — only count violation events, not every frame
            person_count = sum(1 for d in detections if _is_person_detection(d))
            analytics.log_frame(person_count, violations if alert_triggered else [])

            annotated = annotate_frame(processed_frame.copy(), detections, violations)
            _, buffer = cv2.imencode('.jpg', annotated)
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')

            response = {
                "annotated_frame": f"data:image/jpeg;base64,{annotated_base64}",
                "detections": detections,
                "violations": violations,
                "critical_events": critical_events,
                "alert": alert_triggered,
                "response_actions": response_actions,
                "device": device
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("📹 Camera stream stopped")
    except Exception as e:
        print(f"Camera stream error: {e}")
    finally:
        camera.close()

@app.get("/api/analytics")
async def get_analytics():
    """Retrieve safety analytics summary."""
    return analytics.get_summary()

@app.get("/api/violations")
async def get_recorded_violations():
    """List recorded violation evidence."""
    files = sorted(VIOLATIONS_DIR.glob("*.jpg"), reverse=True)
    return {"violations": [f.name for f in files]}

@app.delete("/api/violations")
async def clear_violations():
    """Delete all violation evidence images."""
    files = list(VIOLATIONS_DIR.glob("*.jpg"))
    for f in files:
        f.unlink()
    return {"deleted": len(files)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None,
        "model": loaded_model_ref,
    }

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="GuardianVision Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", default=None, help="Path to YOLO model weights")
    parser.add_argument("--simulate", action="store_true", help="Enable simulation mode for emergency events")
    args = parser.parse_args()
    
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    if args.simulate:
        os.environ["SIMULATION_MODE"] = "true"
        print("🛠️ Simulation Mode ENABLED")
    
    uvicorn.run(app, host=args.host, port=args.port)
