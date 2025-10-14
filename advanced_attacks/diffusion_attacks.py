"""
Diffusion-based attacks for watermark removal.

This module implements attacks using diffusion models (Stable Diffusion, ReSD, etc.)
for watermark removal through image regeneration.
"""

import torch
import numpy as np
from typing import Optional, Union
import logging

from .attack_types import AttackConfig, AttackType

logger = logging.getLogger(__name__)

# Try to import diffusion libraries
try:
    from diffusers import (
        StableDiffusionInpaintPipeline,
        StableDiffusionImg2ImgPipeline,
        AutoPipelineForImage2Image
    )
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    logger.warning("Diffusion models not available. Install with: pip install diffusers transformers accelerate")


class DiffusionAttacker:
    """Implements diffusion-based attacks on watermarked images."""
    
    def __init__(self, device: Union[str, torch.device] = 'cuda'):
        """
        Initialize diffusion attacker.
        
        Args:
            device: Device to use for models
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.models_loaded = False
        
        # Model placeholders
        self.inpaint_model = None
        self.img2img_model = None
        
        if not DIFFUSION_AVAILABLE:
            logger.error("Diffusion libraries not available. Diffusion attacks will not work.")
    
    def _load_models(self):
        """Lazy load diffusion models."""
        if self.models_loaded or not DIFFUSION_AVAILABLE:
            return
        
        logger.info(f"Loading diffusion models on {self.device}...")
        
        try:
            # Use float16 for GPU, float32 for CPU
            dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
            
            # Load inpainting model
            self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=dtype
            ).to(self.device)
            
            # Load img2img model
            self.img2img_model = AutoPipelineForImage2Image.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype
            ).to(self.device)
            
            self.models_loaded = True
            logger.info("Diffusion models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load diffusion models: {str(e)}")
            self.models_loaded = False
    
    def attack(self, image: np.ndarray, config: AttackConfig) -> np.ndarray:
        """
        Apply diffusion-based attack.
        
        Args:
            image: Input image (BGR numpy array)
            config: Attack configuration
            
        Returns:
            Attacked image
        """
        if not DIFFUSION_AVAILABLE:
            logger.error("Diffusion attacks not available")
            return image.copy()
        
        # Load models if needed
        self._load_models()
        
        if not self.models_loaded:
            logger.error("Diffusion models not loaded")
            return image.copy()
        
        # Route to specific attack
        attack_type = config.attack_type
        
        if attack_type == AttackType.DIFFUSION_INPAINTING:
            return self._inpainting_attack(image, **config.params)
        elif attack_type == AttackType.DIFFUSION_REGENERATION:
            return self._regeneration_attack(image, **config.params)
        elif attack_type == AttackType.DIFFUSION_IMG2IMG:
            return self._img2img_attack(image, **config.params)
        else:
            logger.warning(f"Unknown diffusion attack type: {attack_type}")
            return image.copy()
    
    def _inpainting_attack(
        self,
        img: np.ndarray,
        mask_ratio: float = 0.3,
        prompt: str = "A high quality photograph",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> np.ndarray:
        """
        Attack using diffusion inpainting.
        
        This method masks a portion of the image and regenerates it,
        potentially removing watermarks in the masked region.
        """
        # Import here to avoid circular dependency
        from PIL import Image
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from common import image_utils
        
        # Store original dimensions
        original_h, original_w = img.shape[:2]
        
        # Convert to PIL
        img_pil = image_utils.cv2_to_pil(img)
        
        # Ensure dimensions are multiples of 8
        width, height = img_pil.size
        if width % 8 != 0 or height % 8 != 0:
            width = (width // 8) * 8
            height = (height // 8) * 8
            img_pil = img_pil.resize((width, height))
        
        # Create mask (center rectangular mask)
        h, w = height, width
        mask = np.zeros((h, w), dtype=np.uint8)
        
        mask_h, mask_w = int(h * mask_ratio), int(w * mask_ratio)
        start_h, start_w = (h - mask_h) // 2, (w - mask_w) // 2
        mask[start_h:start_h + mask_h, start_w:start_w + mask_w] = 255
        mask_pil = Image.fromarray(mask)
        
        try:
            # Run inpainting
            with torch.no_grad():
                output = self.inpaint_model(
                    prompt=prompt,
                    image=img_pil,
                    mask_image=mask_pil,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]
            
            # Convert back to numpy BGR
            output_cv = image_utils.pil_to_cv2(output)
            
            # Resize to original dimensions
            output_cv = image_utils.ensure_dimensions(output_cv, original_h, original_w)
            
            return output_cv
            
        except Exception as e:
            logger.error(f"Error in inpainting attack: {str(e)}")
            return img
    
    def _regeneration_attack(
        self,
        img: np.ndarray,
        prompt: str = "A high quality photograph",
        strength: float = 0.5,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> np.ndarray:
        """
        Attack using img2img regeneration.
        
        This method uses the image as a starting point and regenerates it
        based on the prompt, potentially removing watermarks.
        """
        return self._img2img_attack(
            img, prompt, strength, num_inference_steps, guidance_scale
        )
    
    def _img2img_attack(
        self,
        img: np.ndarray,
        prompt: str = "A high quality photograph",
        strength: float = 0.5,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> np.ndarray:
        """
        Attack using img2img pipeline.
        """
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from common import image_utils
        
        # Store original dimensions
        original_h, original_w = img.shape[:2]
        
        # Convert to PIL
        img_pil = image_utils.cv2_to_pil(img)
        
        # Ensure dimensions are multiples of 8
        width, height = img_pil.size
        if width % 8 != 0 or height % 8 != 0:
            width = (width // 8) * 8
            height = (height // 8) * 8
            img_pil = img_pil.resize((width, height))
        
        try:
            # Run img2img
            with torch.no_grad():
                output = self.img2img_model(
                    prompt=prompt,
                    image=img_pil,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]
            
            # Convert back to numpy BGR
            output_cv = image_utils.pil_to_cv2(output)
            
            # Resize to original dimensions
            output_cv = image_utils.ensure_dimensions(output_cv, original_h, original_w)
            
            return output_cv
            
        except Exception as e:
            logger.error(f"Error in img2img attack: {str(e)}")
            return img