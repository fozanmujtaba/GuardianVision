"""
WebSocket Test Client for GuardianVision.

Sends sample images to the backend and validates the current /ws protocol:
[4-byte JSON length][JSON metadata][JPEG frame].
"""

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import websockets


def parse_ws_response(response):
    """Parse either the current binary response or a legacy JSON text response."""
    if isinstance(response, str):
        return json.loads(response), b""

    if len(response) < 4:
        raise ValueError("Binary response is too short to contain a JSON length header")

    json_len = int.from_bytes(response[:4], byteorder="big")
    json_end = 4 + json_len
    if len(response) < json_end:
        raise ValueError("Binary response ended before JSON metadata was complete")

    metadata = json.loads(response[4:json_end].decode("utf-8"))
    jpeg_bytes = response[json_end:]
    return metadata, jpeg_bytes


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as f:
        img_data = f.read()
    return f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"


async def test_single_image(ws_url: str, image_path: str):
    """Send a single image and print the response summary."""
    image = Path(image_path)
    img_b64 = encode_image(image)

    print(f"Sending image: {image}")
    start = time.time()

    async with websockets.connect(ws_url) as ws:
        await ws.send(img_b64)
        response = await ws.recv()

    elapsed = (time.time() - start) * 1000
    data, jpeg_bytes = parse_ws_response(response)

    print(f"Response received in {elapsed:.1f}ms")
    print(f"   Detections: {len(data.get('detections', []))}")
    print(f"   Violations: {len(data.get('violations', []))}")
    print(f"   Critical events: {len(data.get('critical_events', []))}")
    print(f"   Annotated JPEG bytes: {len(jpeg_bytes)}")
    print(f"   Alert: {data.get('alert')}")
    print(f"   Device: {data.get('device', 'unknown')}")

    for det in data.get("detections", []):
        print(f"   - {det.get('class_name', det.get('class'))}: {det.get('conf', 0):.2f}")

    return data


async def test_continuous(ws_url: str, image_dir: str, fps: int = 5):
    """Continuously send images from a directory."""
    directory = Path(image_dir)
    images = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png"))
    if not images:
        print(f"No images found in {directory}")
        return

    print(f"Starting continuous test with {len(images)} images at {fps} FPS")
    delay = 1.0 / fps
    idx = 0

    async with websockets.connect(ws_url) as ws:
        while True:
            img_path = images[idx % len(images)]
            img_b64 = encode_image(img_path)

            start = time.time()
            await ws.send(img_b64)
            response = await ws.recv()
            elapsed = (time.time() - start) * 1000

            data, _ = parse_ws_response(response)
            detections = len(data.get("detections", []))
            violations = len(data.get("violations", []))
            print(f"Frame {idx}: {detections} detections, {violations} violations, {elapsed:.1f}ms", end="\r")

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
