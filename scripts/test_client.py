"""
WebSocket Test Client for GuardianVision
Sends sample images to the backend and validates binary responses.

Binary protocol: [4-byte JSON length (big-endian)][JSON bytes][JPEG bytes]
"""

import asyncio
import websockets
import json
import argparse
import time
from pathlib import Path


def parse_binary_response(data: bytes) -> tuple[dict, bytes]:
    """Parse the binary response: [4-byte JSON len][JSON][JPEG]."""
    json_len = int.from_bytes(data[:4], byteorder="big")
    meta = json.loads(data[4 : 4 + json_len])
    jpeg = data[4 + json_len :]
    return meta, jpeg


async def test_single_image(ws_url: str, image_path: str):
    """Send a single image and print the response."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    print(f"📤 Sending image: {image_path} ({len(img_bytes)} bytes)")
    start = time.time()

    async with websockets.connect(ws_url) as ws:
        await ws.send(img_bytes)
        response = await ws.recv()

    elapsed = (time.time() - start) * 1000

    if isinstance(response, bytes):
        data, jpeg = parse_binary_response(response)
        print(f"✅ Response received in {elapsed:.1f}ms (JPEG: {len(jpeg)} bytes)")
    else:
        data = json.loads(response)
        print(f"✅ Response received in {elapsed:.1f}ms (text)")

    print(f"   Detections : {len(data.get('detections', []))}")
    print(f"   Violations : {len(data.get('violations', []))}")
    print(f"   Alert      : {data.get('alert')}")
    print(f"   Device     : {data.get('device', 'unknown')}")

    for det in data.get("detections", []):
        print(f"   - {det.get('class_name', det['class'])}: {det['conf']:.2f}")

    return data


async def test_continuous(ws_url: str, image_dir: str, fps: int = 5):
    """Continuously send images from a directory."""
    images = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    if not images:
        print(f"❌ No images found in {image_dir}")
        return

    print(f"🔄 Continuous test: {len(images)} images at {fps} FPS")
    delay = 1.0 / fps
    idx = 0

    async with websockets.connect(ws_url) as ws:
        while True:
            img_path = images[idx % len(images)]
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            start = time.time()
            await ws.send(img_bytes)
            response = await ws.recv()
            elapsed = (time.time() - start) * 1000

            if isinstance(response, bytes):
                data, _ = parse_binary_response(response)
            else:
                data = json.loads(response)

            n_det = len(data.get("detections", []))
            n_vio = len(data.get("violations", []))
            print(f"Frame {idx:4d}: {n_det} detections  {n_vio} violations  {elapsed:.1f}ms", end="\r")

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
