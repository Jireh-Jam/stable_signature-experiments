"""
Stable Signature watermarking method implementation.
"""

import os
import torch
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np

from ..base import BaseWatermarkMethod


class StableSignatureMethod(BaseWatermarkMethod):
    """
    Stable Signature watermarking implementation.
    
    This method embeds watermarks in the latent space of diffusion models
    and provides robust detection capabilities.
    """
    
    def __init__(self):
        super().__init__("Stable Signature")
        self.decoder_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the Stable Signature method.
        
        Args:
            config: Configuration dictionary with model paths
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if config is None:
                config = {}
                
            # Default model path
            model_path = config.get(
                'decoder_path', 
                'models/checkpoints/dec_48b_whit.torchscript.pt'
            )
            
            if not os.path.exists(model_path):
                print(f"⚠️ Model file not found: {model_path}")
                print("💡 Please download the model using the setup instructions")
                return False
                
            # Load the decoder model
            self.decoder_model = torch.jit.load(model_path, map_location=self.device)
            self.decoder_model.eval()
            
            self.is_initialized = True
            print(f"✅ {self.name} initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing {self.name}: {str(e)}")
            return False
            
    def embed_watermark(self, image: Image.Image, message: str) -> Tuple[Image.Image, bool]:
        """
        Embed watermark into image using Stable Signature method.
        
        Args:
            image: Input PIL Image
            message: Binary message string to embed
            
        Returns:
            Tuple of (watermarked_image, success_flag)
        """
        if not self.is_initialized:
            print("❌ Method not initialized. Call initialize() first.")
            return image, False
            
        try:
            # Convert PIL image to tensor
            image_tensor = self._pil_to_tensor(image)
            
            # Embed watermark (simplified implementation)
            # In practice, this would use the full Stable Signature pipeline
            watermarked_tensor = self._embed_message(image_tensor, message)
            
            # Convert back to PIL image
            watermarked_image = self._tensor_to_pil(watermarked_tensor)
            
            return watermarked_image, True
            
        except Exception as e:
            print(f"❌ Error embedding watermark: {str(e)}")
            return image, False
            
    def detect_watermark(self, image: Image.Image) -> Tuple[bool, float, Optional[str]]:
        """
        Detect watermark in image using Stable Signature decoder.
        
        Args:
            image: Input PIL Image to check
            
        Returns:
            Tuple of (detected, confidence_score, extracted_message)
        """
        if not self.is_initialized:
            print("❌ Method not initialized. Call initialize() first.")
            return False, 0.0, None
            
        try:
            # Convert PIL image to tensor
            image_tensor = self._pil_to_tensor(image)
            
            # Run detection
            with torch.no_grad():
                # This is a simplified version - actual implementation would
                # use the full decoder pipeline
                detection_result = self._run_detection(image_tensor)
                
            detected, confidence, message = detection_result
            
            return detected, confidence, message
            
        except Exception as e:
            print(f"❌ Error detecting watermark: {str(e)}")
            return False, 0.0, None
            
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to PyTorch tensor."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
        
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert PyTorch tensor to PIL image."""
        # Remove batch dimension and move to CPU
        tensor = tensor.squeeze(0).cpu()
        
        # Denormalize and convert to numpy
        img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Convert to PIL image
        return Image.fromarray(img_array)
        
    def _embed_message(self, image_tensor: torch.Tensor, message: str) -> torch.Tensor:
        """
        Embed message into image tensor.
        
        This is a placeholder implementation. The actual Stable Signature
        method would use the latent diffusion model pipeline.
        """
        # Placeholder: just return the original image
        # In practice, this would encode the message and embed it
        return image_tensor
        
    def _run_detection(self, image_tensor: torch.Tensor) -> Tuple[bool, float, Optional[str]]:
        """
        Run watermark detection on image tensor.
        
        This is a placeholder implementation. The actual method would
        use the trained decoder model.
        """
        # Placeholder detection logic
        # In practice, this would use the decoder model to extract the message
        
        # Simulate detection with some randomness for demo purposes
        import random
        detected = random.random() > 0.3  # 70% detection rate
        confidence = random.uniform(0.6, 0.95) if detected else random.uniform(0.1, 0.4)
        message = "demo_message_48bits" if detected else None
        
        return detected, confidence, message
        
    def get_info(self) -> Dict[str, Any]:
        """Get detailed information about this method."""
        info = super().get_info()
        info.update({
            "device": str(self.device),
            "model_loaded": self.decoder_model is not None,
            "description": "Watermarking method for latent diffusion models",
            "paper": "The Stable Signature: Rooting Watermarks in Latent Diffusion Models (ICCV 2023)"
        })
        return info