"""
Watermark Anything method implementation.
"""

from typing import Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np

from ..base import BaseWatermarkMethod
from .backend import WAMBackend


class WatermarkAnythingMethod(BaseWatermarkMethod):
    """
    Watermark Anything implementation.
    
    This method provides a general-purpose watermarking approach.
    """
    
    def __init__(self):
        super().__init__("Watermark Anything")
        self.backend: Optional[WAMBackend] = None
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the Watermark Anything method."""
        try:
            self.backend = WAMBackend(config=config)
            ok = self.backend.initialize()
            self.is_initialized = bool(ok)
            if ok:
                print(f"✅ {self.name} initialized successfully")
            else:
                print(f"⚠️ {self.name} initialization failed; using graceful fallback")
            return bool(ok)
        except Exception as e:
            print(f"❌ Error initializing {self.name}: {str(e)}")
            self.is_initialized = False
            return False
            
    def embed_watermark(self, image: Image.Image, message: str) -> Tuple[Image.Image, bool]:
        """Embed watermark using Watermark Anything method."""
        if not self.is_initialized or self.backend is None:
            return image, False

        # Delegate to backend (will gracefully no-op if real model unavailable)
        return self.backend.embed(image, message)
        
    def detect_watermark(self, image: Image.Image) -> Tuple[bool, float, Optional[str]]:
        """Detect watermark using Watermark Anything method."""
        if not self.is_initialized or self.backend is None:
            return False, 0.0, None

        # Delegate to backend (will gracefully simulate if real model unavailable)
        return self.backend.detect(image)