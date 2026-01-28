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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from ultralytics import YOLO
from auditor import PPEAuditor
from camera import CameraStream, FrameSimulator
from analytics import AnalyticsManager
from response_manager import response_manager

# Configuration (Defaults)
DEFAULT_MODEL_PATH = "../models/ppe_model.pt"
FALLBACK_MODEL = "yolo11n.pt"

# Global state
model = None
device = None
auditor = None
camera_task = None
analytics = None
SIMULATION_MODE = os.environ.get("SIMULATION_MODE", "false").lower() == "true"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    global model, device, auditor, analytics
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if os.environ.get("FORCE_CPU", "false").lower() == "true":
        device = "cpu"
    print(f"🚀 Device: {device}")
    
    # Determine model path dynamically
    current_model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    
    # Load custom PPE model if available, otherwise fallback
    if os.path.exists(current_model_path):
        print(f"📦 Loading custom safety model: {current_model_path}")
        model = YOLO(current_model_path)
    else:
        print(f"⚠️ Custom model not found at {current_model_path}, using fallback: {FALLBACK_MODEL}")
        model = YOLO(FALLBACK_MODEL)
    
    model.to(device)
    auditor = PPEAuditor(cooldown_seconds=10)
    analytics = AnalyticsManager()
    
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
app.mount("/violations", StaticFiles(directory="violations"), name="violations")

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

# Comprehensive Safety Classes (24 Classes)
CLASS_NAMES = {
    0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask", 4: "NO-Safety Vest", 
    5: "Person", 6: "Safety Cone", 7: "Safety Vest", 8: "machinery", 9: "vehicle",
    10: "Fire", 11: "Smoke", 12: "Emergency Exit Sign", 13: "Fire Extinguisher", 
    14: "Fall Detected", 15: "Sitting", 16: "Fire Blanket", 17: "Manual Call Point", 
    18: "Smoke Detector", 19: "Wall Hydrant Sign", 20: "Fire Extinguisher Sign Old", 
    21: "Call Point Sign", 22: "Fire Door Sign", 23: "Fire Extinguisher Sign"
}

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
        
        if det["class"] == 5:  # Person
            color = (255, 200, 0)  # Cyan for person
        elif det["class"] in [10, 11, 14]: # Fire, Smoke, Fall
            color = (0, 0, 255) # Red for critical
        elif det["class"] in [0, 7]: # Hardhat, Safety Vest
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
            data = await websocket.receive_text()
            frame_count += 1
            
            if frame_count == 1:
                print(f"📸 First frame received! Data length: {len(data)}")
            
            # Process every 2nd frame to reduce lag (skip odd frames)
            if frame_count % 2 != 0:
                continue
            
            if frame_count % 30 == 0:
                print(f"📦 Received frame ${frame_count}")
            
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
                await websocket.send_text(json.dumps({"error": "Decode error", "details": str(e)}))
                continue
            
            if frame is None:
                continue

            processed_frame = preprocess_frame(frame)
            
            # Run inference with tracking (BoTSORT) - Optimizing imgsz for speed
            inf_start = time.time()
            results = model.track(processed_frame, device=device, persist=True, verbose=False, imgsz=480)
            inf_time = (time.time() - inf_start) * 1000
            
            if frame_count % 30 == 0:
                print(f"⏱️ Inference process {frame_count}: {inf_time:.2f}ms")
            
            detections = process_detections(results)
            
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
            
            # Log analytics
            person_count = len([d for d in detections if d['class'] == 5])
            analytics.log_frame(person_count, violations)
            
            # Annotate frame
            annotated = annotate_frame(processed_frame.copy(), detections, violations)
            
            # Encode response
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
            
            # Log analytics
            person_count = len([d for d in detections if d['class'] == 5])
            analytics.log_frame(person_count, violations)
            
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
    import glob
    files = glob.glob("violations/*.jpg")
    return {"violations": [os.path.basename(f) for f in files]}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
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
