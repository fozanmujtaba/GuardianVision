"""
Camera capture module for GuardianVision.
Provides async webcam frame generation for WebSocket streaming.
"""

import cv2
import asyncio
import base64
from typing import AsyncGenerator, Optional, Union

class CameraStream:
    """
    Async webcam/video capture class.
    """
    
    def __init__(self, source: Union[int, str] = 0, fps: int = 30):
        """
        Initialize camera stream.
        
        Args:
            source: Camera index (0 for default) or video file path
            fps: Target frames per second
        """
        self.source = source
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_delay = 1.0 / fps
        
    def open(self) -> bool:
        """Open the camera/video source."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera source: {self.source}")
            return False
        
        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        print(f"✅ Camera opened: {self.source}")
        return True
    
    def close(self):
        """Release the camera."""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def read_frame(self) -> Optional[bytes]:
        """Read a single frame and return as base64 JPEG."""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    async def stream_frames(self) -> AsyncGenerator[str, None]:
        """
        Async generator that yields base64-encoded frames.
        """
        if not self.open():
            return
        
        try:
            while True:
                frame_b64 = self.read_frame()
                if frame_b64 is None:
                    break
                
                yield f"data:image/jpeg;base64,{frame_b64}"
                await asyncio.sleep(self.frame_delay)
        finally:
            self.close()


class FrameSimulator:
    """
    Simulates video frames for testing without a camera.
    Loads images from a directory and cycles through them.
    """
    
    def __init__(self, image_dir: str, fps: int = 10):
        self.image_dir = image_dir
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self.images: list = []
        self.current_idx = 0
        
    def load_images(self):
        """Load all images from the directory."""
        import os
        import glob
        
        patterns = ['*.jpg', '*.jpeg', '*.png']
        for pattern in patterns:
            self.images.extend(glob.glob(os.path.join(self.image_dir, pattern)))
        
        self.images.sort()
        print(f"📷 Loaded {len(self.images)} test images")
        
    async def stream_frames(self) -> AsyncGenerator[str, None]:
        """Yield frames in a loop."""
        self.load_images()
        
        if not self.images:
            print("❌ No images found in directory")
            return
        
        while True:
            img_path = self.images[self.current_idx]
            frame = cv2.imread(img_path)
            
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                yield f"data:image/jpeg;base64,{frame_b64}"
            
            self.current_idx = (self.current_idx + 1) % len(self.images)
            await asyncio.sleep(self.frame_delay)
