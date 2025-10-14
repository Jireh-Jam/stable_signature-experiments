"""
Main watermark attack implementation module.

This module provides the WatermarkAttacker class that implements various
attack methods for testing watermark robustness.
"""

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from typing import Optional, Dict, Any, Tuple, Union, List
import logging
import tempfile
import os
from pathlib import Path

# Import common utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from common import image_utils, metrics

from .attack_types import AttackType, AttackConfig
from .frequency_attacks import FrequencyAttacker
from .diffusion_attacks import DiffusionAttacker

logger = logging.getLogger(__name__)


class WatermarkAttacker:
    """
    Main class for watermark attacks.
    
    This class provides various attack methods to test watermark robustness,
    including traditional image processing and advanced AI-based attacks.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the watermark attacker.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initialized WatermarkAttacker on device: {self.device}")
        
        # Initialize specialized attackers
        self.frequency_attacker = FrequencyAttacker()
        self.diffusion_attacker = None  # Lazy initialization
    
    def attack(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        config: AttackConfig
    ) -> np.ndarray:
        """
        Apply an attack to an image.
        
        Args:
            image: Input image (numpy array BGR, PIL Image, or tensor)
            config: Attack configuration
            
        Returns:
            Attacked image as numpy array (BGR format)
        """
        # Convert input to numpy BGR format
        if isinstance(image, torch.Tensor):
            image = image_utils.tensor_to_image(image)
            image = image_utils.pil_to_cv2(image)
        elif isinstance(image, Image.Image):
            image = image_utils.pil_to_cv2(image)
        
        # Store original dimensions
        original_h, original_w = image.shape[:2]
        
        # Apply attack based on type
        attack_type = config.attack_type
        params = config.params
        
        logger.info(f"Applying attack: {config.description or attack_type.value}")
        
        try:
            # Basic image processing attacks
            if attack_type == AttackType.GAUSSIAN_BLUR:
                result = self._gaussian_blur(image, **params)
            elif attack_type == AttackType.GAUSSIAN_NOISE:
                result = self._gaussian_noise(image, **params)
            elif attack_type == AttackType.JPEG_COMPRESSION:
                result = self._jpeg_compression(image, **params)
            elif attack_type == AttackType.BRIGHTNESS:
                result = self._brightness(image, **params)
            elif attack_type == AttackType.CONTRAST:
                result = self._contrast(image, **params)
            elif attack_type == AttackType.ROTATION:
                result = self._rotation(image, **params)
            elif attack_type == AttackType.SCALE:
                result = self._scale(image, **params)
            elif attack_type == AttackType.CROP:
                result = self._crop(image, **params)
            
            # Advanced attacks
            elif attack_type == AttackType.HIGH_FREQUENCY:
                result = self.frequency_attacker.high_frequency_attack(image, **params)
            
            # Diffusion attacks (lazy initialization)
            elif attack_type in [AttackType.DIFFUSION_INPAINTING, 
                               AttackType.DIFFUSION_REGENERATION,
                               AttackType.DIFFUSION_IMG2IMG,
                               AttackType.DIFFUSION_RESD]:
                if self.diffusion_attacker is None:
                    self._init_diffusion_attacker()
                result = self.diffusion_attacker.attack(image, config)
            
            else:
                logger.warning(f"Unknown attack type: {attack_type}")
                result = image.copy()
            
            # Ensure output has original dimensions
            result = image_utils.ensure_dimensions(result, original_h, original_w)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in attack {attack_type}: {str(e)}")
            return image.copy()
    
    def _init_diffusion_attacker(self):
        """Initialize diffusion attacker (lazy loading)."""
        try:
            self.diffusion_attacker = DiffusionAttacker(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize diffusion attacker: {str(e)}")
            self.diffusion_attacker = None
    
    # Basic attack implementations
    
    def _gaussian_blur(self, img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur."""
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    def _gaussian_noise(self, img: np.ndarray, std: float = 0.05) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, std * 255, img.shape)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    def _jpeg_compression(self, img: np.ndarray, quality: int = 80) -> np.ndarray:
        """Apply JPEG compression."""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Convert to PIL and save with compression
            pil_img = image_utils.cv2_to_pil(img)
            pil_img.save(temp_path, 'JPEG', quality=quality)
            
            # Read back
            compressed = cv2.imread(temp_path)
            return compressed
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _brightness(self, img: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Adjust brightness."""
        pil_img = image_utils.cv2_to_pil(img)
        enhancer = ImageEnhance.Brightness(pil_img)
        enhanced = enhancer.enhance(factor)
        return image_utils.pil_to_cv2(enhanced)
    
    def _contrast(self, img: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Adjust contrast."""
        pil_img = image_utils.cv2_to_pil(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(factor)
        return image_utils.pil_to_cv2(enhanced)
    
    def _rotation(self, img: np.ndarray, degrees: float = 15.0) -> np.ndarray:
        """Rotate image."""
        pil_img = image_utils.cv2_to_pil(img)
        rotated = pil_img.rotate(degrees, expand=True, fillcolor='white')
        return image_utils.pil_to_cv2(rotated)
    
    def _scale(self, img: np.ndarray, factor: float = 0.5) -> np.ndarray:
        """Scale image."""
        h, w = img.shape[:2]
        new_size = (int(w * factor), int(h * factor))
        
        # Scale down
        scaled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        
        # Scale back up to original size
        return cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def _crop(self, img: np.ndarray, ratio: float = 0.5) -> np.ndarray:
        """Crop and rescale image."""
        h, w = img.shape[:2]
        crop_h, crop_w = int(h * ratio), int(w * ratio)
        
        # Center crop
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        cropped = img[start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        # Resize back to original
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def evaluate_attack(
        self,
        original: np.ndarray,
        watermarked: np.ndarray,
        config: AttackConfig,
        watermark_msg: Optional[np.ndarray] = None,
        decoded_msg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply attack and evaluate its effectiveness.
        
        Args:
            original: Original image without watermark
            watermarked: Watermarked image
            config: Attack configuration
            watermark_msg: Original watermark message (optional)
            decoded_msg: Decoded message after attack (optional)
            
        Returns:
            Tuple of (attacked image, metrics dictionary)
        """
        # Apply attack
        attacked = self.attack(watermarked, config)
        
        # Calculate metrics
        attack_metrics = metrics.comprehensive_attack_metrics(
            original, watermarked, attacked,
            watermark_msg, decoded_msg
        )
        
        # Add attack info
        attack_metrics['attack_type'] = config.attack_type.value
        attack_metrics['attack_description'] = config.description
        attack_metrics['attack_params'] = config.params
        
        return attacked, attack_metrics
    
    def batch_evaluate(
        self,
        image_pairs: List[Tuple[np.ndarray, np.ndarray]],
        attack_configs: Dict[str, AttackConfig],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple attacks on multiple image pairs.
        
        Args:
            image_pairs: List of (original, watermarked) image pairs
            attack_configs: Dictionary of attack configurations
            output_dir: Optional directory to save results
            
        Returns:
            List of all metrics
        """
        all_metrics = []
        
        for i, (original, watermarked) in enumerate(image_pairs):
            logger.info(f"Processing image pair {i+1}/{len(image_pairs)}")
            
            for attack_name, config in attack_configs.items():
                logger.info(f"Applying {attack_name}")
                
                try:
                    attacked, metrics_dict = self.evaluate_attack(
                        original, watermarked, config
                    )
                    
                    # Add identifiers
                    metrics_dict['image_idx'] = i
                    metrics_dict['attack_name'] = attack_name
                    
                    all_metrics.append(metrics_dict)
                    
                    # Save attacked image if output directory provided
                    if output_dir:
                        output_path = Path(output_dir) / f"img{i}_{attack_name}.png"
                        cv2.imwrite(str(output_path), attacked)
                        
                except Exception as e:
                    logger.error(f"Error in batch evaluation: {str(e)}")
                    continue
        
        return all_metrics