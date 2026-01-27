"""
WebSocket Test Client for GuardianVision
Sends sample images to the backend and validates responses.
"""

import asyncio
import websockets
import base64
import json
import argparse
import os
import time
from pathlib import Path

async def test_single_image(ws_url: str, image_path: str):
    """Send a single image and print the response."""
    
    # Read and encode image
    with open(image_path, "rb") as f:
        img_data = f.read()
    
    img_b64 = f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"
    
    print(f"📤 Sending image: {image_path}")
    start = time.time()
    
    async with websockets.connect(ws_url) as ws:
        await ws.send(img_b64)
        response = await ws.recv()
        
    elapsed = (time.time() - start) * 1000
    data = json.loads(response)
    
    print(f"✅ Response received in {elapsed:.1f}ms")
    print(f"   Detections: {len(data['detections'])}")
    print(f"   Violations: {len(data['violations'])}")
    print(f"   Alert: {data['alert']}")
    print(f"   Device: {data.get('device', 'unknown')}")
    
    # Print detection details
    for det in data['detections']:
        print(f"   - {det.get('class_name', det['class'])}: {det['conf']:.2f}")
    
    return data

async def test_continuous(ws_url: str, image_dir: str, fps: int = 5):
    """Continuously send images from a directory."""
    
    images = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    if not images:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"🔄 Starting continuous test with {len(images)} images at {fps} FPS")
    delay = 1.0 / fps
    idx = 0
    
    async with websockets.connect(ws_url) as ws:
        while True:
            img_path = images[idx % len(images)]
            
            with open(img_path, "rb") as f:
                img_data = f.read()
            
            img_b64 = f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"
            
            start = time.time()
            await ws.send(img_b64)
            response = await ws.recv()
            elapsed = (time.time() - start) * 1000
            
            data = json.loads(response)
            print(f"Frame {idx}: {len(data['detections'])} detections, {elapsed:.1f}ms", end="\r")
            
            idx += 1
            await asyncio.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description="GuardianVision Test Client")
    parser.add_argument("--url", default="ws://localhost:8000/ws", help="WebSocket URL")
    parser.add_argument("--image", help="Path to a single test image")
    parser.add_argument("--dir", help="Directory of images for continuous testing")
    parser.add_argument("--fps", type=int, default=5, help="FPS for continuous testing")
    args = parser.parse_args()
    
    if args.image:
        asyncio.run(test_single_image(args.url, args.image))
    elif args.dir:
        asyncio.run(test_continuous(args.url, args.dir, args.fps))
    else:
        print("Usage: python test_client.py --image <path> OR --dir <path>")

if __name__ == "__main__":
    main()
