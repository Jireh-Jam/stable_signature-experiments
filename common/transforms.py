"""
Unified transformation pipeline for watermark robustness testing.

This module consolidates image transformations from multiple sources and provides
a registry-based system for applying transformations with proper configuration
and error handling.
"""

import logging
from typing import Dict, Callable, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import tempfile
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import functional as F
from skimage.util import random_noise

from .image_utils import NORMALIZE_IMAGENET, UNNORMALIZE_IMAGENET

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """Result of applying a transformation."""
    image: Image.Image
    parameters: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class BaseTransform(ABC):
    """Base class for all image transformations."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def apply(self, image: Image.Image, **kwargs) -> TransformResult:
        """Apply the transformation to an image."""
        pass
    
    def __call__(self, image: Image.Image, **kwargs) -> TransformResult:
        """Make the transform callable."""
        try:
            result = self.apply(image, **kwargs)
            logger.debug(f"Applied transform '{self.name}' with params: {result.parameters}")
            return result
        except Exception as e:
            logger.error(f"Transform '{self.name}' failed: {str(e)}")
            return TransformResult(
                image=image,
                parameters=kwargs,
                success=False,
                error_message=str(e)
            )


class GeometricTransforms:
    """Collection of geometric transformations."""
    
    class CropCenter(BaseTransform):
        def apply(self, image: Image.Image, crop_percentage: float = 0.1) -> TransformResult:
            width, height = image.size
            crop_w = int(width * crop_percentage)
            crop_h = int(height * crop_percentage)
            
            left = crop_w
            top = crop_h
            right = width - crop_w
            bottom = height - crop_h
            
            result_image = image.crop((left, top, right, bottom))
            return TransformResult(
                image=result_image,
                parameters={'crop_percentage': crop_percentage},
                success=True
            )
    
    class CropRandom(BaseTransform):
        def apply(self, image: Image.Image, crop_percentage: float = 0.1) -> TransformResult:
            width, height = image.size
            crop_w = int(width * crop_percentage)
            crop_h = int(height * crop_percentage)
            
            left = random.randint(0, crop_w)
            top = random.randint(0, crop_h)
            right = width - random.randint(0, crop_w)
            bottom = height - random.randint(0, crop_h)
            
            result_image = image.crop((left, top, right, bottom))
            return TransformResult(
                image=result_image,
                parameters={'crop_percentage': crop_percentage, 'left': left, 'top': top},
                success=True
            )
    
    class Resize(BaseTransform):
        def apply(self, image: Image.Image, scale: float = 0.8) -> TransformResult:
            width, height = image.size
            new_size = (int(width * scale), int(height * scale))
            result_image = image.resize(new_size, Image.Resampling.LANCZOS)
            return TransformResult(
                image=result_image,
                parameters={'scale': scale, 'new_size': new_size},
                success=True
            )
    
    class Rotate(BaseTransform):
        def apply(self, image: Image.Image, degrees: float = 5.0) -> TransformResult:
            result_image = image.rotate(degrees, expand=True, fillcolor='white')
            return TransformResult(
                image=result_image,
                parameters={'degrees': degrees},
                success=True
            )
    
    class Perspective(BaseTransform):
        def apply(self, image: Image.Image, distortion_scale: float = 0.5) -> TransformResult:
            transform = transforms.RandomPerspective(distortion_scale=distortion_scale, p=1.0)
            result_image = transform(image)
            return TransformResult(
                image=result_image,
                parameters={'distortion_scale': distortion_scale},
                success=True
            )


class FilterTransforms:
    """Collection of filtering transformations."""
    
    class GaussianBlur(BaseTransform):
        def apply(self, image: Image.Image, radius: float = 1.0) -> TransformResult:
            result_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            return TransformResult(
                image=result_image,
                parameters={'radius': radius},
                success=True
            )
    
    class MotionBlur(BaseTransform):
        def apply(self, image: Image.Image, size: int = 5) -> TransformResult:
            kernel = ImageFilter.Kernel(
                (size, size), 
                [1/size] * size + [0] * (size * size - size)
            )
            result_image = image.filter(kernel)
            return TransformResult(
                image=result_image,
                parameters={'size': size},
                success=True
            )
    
    class Sharpen(BaseTransform):
        def apply(self, image: Image.Image, factor: float = 2.0) -> TransformResult:
            # Convert to tensor for sharpness adjustment
            tensor = transforms.ToTensor()(image)
            adjusted = F.adjust_sharpness(tensor, factor)
            result_image = transforms.ToPILImage()(adjusted)
            return TransformResult(
                image=result_image,
                parameters={'factor': factor},
                success=True
            )


class ColorTransforms:
    """Collection of color and brightness transformations."""
    
    class Brightness(BaseTransform):
        def apply(self, image: Image.Image, factor: float = 1.2) -> TransformResult:
            enhancer = ImageEnhance.Brightness(image)
            result_image = enhancer.enhance(factor)
            return TransformResult(
                image=result_image,
                parameters={'factor': factor},
                success=True
            )
    
    class Contrast(BaseTransform):
        def apply(self, image: Image.Image, factor: float = 1.2) -> TransformResult:
            enhancer = ImageEnhance.Contrast(image)
            result_image = enhancer.enhance(factor)
            return TransformResult(
                image=result_image,
                parameters={'factor': factor},
                success=True
            )
    
    class Saturation(BaseTransform):
        def apply(self, image: Image.Image, factor: float = 1.2) -> TransformResult:
            enhancer = ImageEnhance.Color(image)
            result_image = enhancer.enhance(factor)
            return TransformResult(
                image=result_image,
                parameters={'factor': factor},
                success=True
            )
    
    class Hue(BaseTransform):
        def apply(self, image: Image.Image, hue_factor: float = 0.1) -> TransformResult:
            tensor = transforms.ToTensor()(image)
            adjusted = F.adjust_hue(tensor, hue_factor)
            result_image = transforms.ToPILImage()(adjusted)
            return TransformResult(
                image=result_image,
                parameters={'hue_factor': hue_factor},
                success=True
            )
    
    class Gamma(BaseTransform):
        def apply(self, image: Image.Image, gamma: float = 2.0, gain: float = 1.0) -> TransformResult:
            tensor = transforms.ToTensor()(image)
            adjusted = F.adjust_gamma(tensor, gamma, gain)
            result_image = transforms.ToPILImage()(adjusted)
            return TransformResult(
                image=result_image,
                parameters={'gamma': gamma, 'gain': gain},
                success=True
            )
    
    class Grayscale(BaseTransform):
        def apply(self, image: Image.Image) -> TransformResult:
            transform = transforms.Grayscale(num_output_channels=3)
            result_image = transform(image)
            return TransformResult(
                image=result_image,
                parameters={},
                success=True
            )


class NoiseTransforms:
    """Collection of noise-based transformations."""
    
    class GaussianNoise(BaseTransform):
        def apply(self, image: Image.Image, noise_level: float = 0.1) -> TransformResult:
            img_array = np.array(image).astype(np.float32)
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            result_image = Image.fromarray(noisy_array)
            return TransformResult(
                image=result_image,
                parameters={'noise_level': noise_level},
                success=True
            )
    
    class SaltPepperNoise(BaseTransform):
        def apply(self, image: Image.Image, amount: float = 0.05) -> TransformResult:
            img_array = np.array(image).astype(np.float32) / 255.0
            noisy_array = random_noise(img_array, mode='s&p', amount=amount)
            result_array = (np.clip(noisy_array, 0, 1) * 255).astype(np.uint8)
            result_image = Image.fromarray(result_array)
            return TransformResult(
                image=result_image,
                parameters={'amount': amount},
                success=True
            )
    
    class RandomErasing(BaseTransform):
        def apply(self, image: Image.Image, 
                 scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3)) -> TransformResult:
            tensor = transforms.ToTensor()(image)
            transform = transforms.RandomErasing(p=1.0, scale=scale, ratio=ratio)
            erased_tensor = transform(tensor)
            result_image = transforms.ToPILImage()(erased_tensor)
            return TransformResult(
                image=result_image,
                parameters={'scale': scale, 'ratio': ratio},
                success=True
            )


class CompressionTransforms:
    """Collection of compression-based transformations."""
    
    class JPEGCompression(BaseTransform):
        def apply(self, image: Image.Image, quality: int = 85) -> TransformResult:
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            result_image = Image.open(buffer).copy()
            return TransformResult(
                image=result_image,
                parameters={'quality': quality},
                success=True
            )
    
    class BitMask(BaseTransform):
        def apply(self, image: Image.Image, bits: int = 3) -> TransformResult:
            img_array = np.array(image)
            mask = 0xFF << bits
            masked_array = img_array & mask
            result_image = Image.fromarray(masked_array)
            return TransformResult(
                image=result_image,
                parameters={'bits': bits},
                success=True
            )


class OverlayTransforms:
    """Collection of overlay-based transformations."""
    
    class TextOverlay(BaseTransform):
        def apply(self, image: Image.Image, 
                 text: str = 'Sample Text',
                 position: Tuple[int, int] = (50, 50),
                 color: Tuple[int, int, int] = (255, 255, 255),
                 font_size: int = 20) -> TransformResult:
            result_image = image.copy()
            draw = ImageDraw.Draw(result_image)
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text(position, text, fill=color, font=font)
            return TransformResult(
                image=result_image,
                parameters={'text': text, 'position': position, 'color': color, 'font_size': font_size},
                success=True
            )


class TransformRegistry:
    """Registry for managing and applying transformations."""
    
    def __init__(self):
        self._transforms: Dict[str, BaseTransform] = {}
        self._register_default_transforms()
    
    def _register_default_transforms(self):
        """Register all default transformations."""
        # Geometric transforms
        self.register('crop_center', GeometricTransforms.CropCenter('crop_center', 'Center crop transformation'))
        self.register('crop_random', GeometricTransforms.CropRandom('crop_random', 'Random crop transformation'))
        self.register('resize', GeometricTransforms.Resize('resize', 'Resize transformation'))
        self.register('rotate', GeometricTransforms.Rotate('rotate', 'Rotation transformation'))
        self.register('perspective', GeometricTransforms.Perspective('perspective', 'Perspective transformation'))
        
        # Filter transforms
        self.register('gaussian_blur', FilterTransforms.GaussianBlur('gaussian_blur', 'Gaussian blur filter'))
        self.register('motion_blur', FilterTransforms.MotionBlur('motion_blur', 'Motion blur filter'))
        self.register('sharpen', FilterTransforms.Sharpen('sharpen', 'Sharpening filter'))
        
        # Color transforms
        self.register('brightness', ColorTransforms.Brightness('brightness', 'Brightness adjustment'))
        self.register('contrast', ColorTransforms.Contrast('contrast', 'Contrast adjustment'))
        self.register('saturation', ColorTransforms.Saturation('saturation', 'Saturation adjustment'))
        self.register('hue', ColorTransforms.Hue('hue', 'Hue adjustment'))
        self.register('gamma', ColorTransforms.Gamma('gamma', 'Gamma correction'))
        self.register('grayscale', ColorTransforms.Grayscale('grayscale', 'Grayscale conversion'))
        
        # Noise transforms
        self.register('gaussian_noise', NoiseTransforms.GaussianNoise('gaussian_noise', 'Gaussian noise'))
        self.register('salt_pepper_noise', NoiseTransforms.SaltPepperNoise('salt_pepper_noise', 'Salt and pepper noise'))
        self.register('random_erasing', NoiseTransforms.RandomErasing('random_erasing', 'Random erasing'))
        
        # Compression transforms
        self.register('jpeg_compression', CompressionTransforms.JPEGCompression('jpeg_compression', 'JPEG compression'))
        self.register('bit_mask', CompressionTransforms.BitMask('bit_mask', 'Bit masking'))
        
        # Overlay transforms
        self.register('text_overlay', OverlayTransforms.TextOverlay('text_overlay', 'Text overlay'))
    
    def register(self, name: str, transform: BaseTransform):
        """Register a new transformation."""
        self._transforms[name] = transform
        logger.debug(f"Registered transform: {name}")
    
    def get_transform(self, name: str) -> Optional[BaseTransform]:
        """Get a transformation by name."""
        return self._transforms.get(name)
    
    def list_transforms(self) -> List[str]:
        """List all available transformation names."""
        return list(self._transforms.keys())
    
    def apply_transform(self, image: Image.Image, transform_name: str, **kwargs) -> TransformResult:
        """Apply a transformation by name."""
        transform = self.get_transform(transform_name)
        if transform is None:
            return TransformResult(
                image=image,
                parameters=kwargs,
                success=False,
                error_message=f"Transform '{transform_name}' not found"
            )
        
        return transform(image, **kwargs)
    
    def apply_transform_chain(self, image: Image.Image, 
                            transforms_config: List[Tuple[str, Dict[str, Any]]]) -> List[TransformResult]:
        """
        Apply a chain of transformations.
        
        Args:
            image: Input image
            transforms_config: List of (transform_name, parameters) tuples
            
        Returns:
            List of TransformResult objects
        """
        results = []
        current_image = image
        
        for transform_name, params in transforms_config:
            result = self.apply_transform(current_image, transform_name, **params)
            results.append(result)
            
            if result.success:
                current_image = result.image
            else:
                logger.warning(f"Transform '{transform_name}' failed, continuing with previous image")
        
        return results


# Global registry instance
transform_registry = TransformRegistry()


def get_standard_transform_configs() -> Dict[str, Dict[str, Any]]:
    """Get standard transformation configurations for testing."""
    return {
        # Geometric
        'crop_10_percent': {'crop_percentage': 0.1},
        'crop_20_percent': {'crop_percentage': 0.2},
        'resize_80': {'scale': 0.8},
        'resize_60': {'scale': 0.6},
        'rotate_5': {'degrees': 5.0},
        'rotate_15': {'degrees': 15.0},
        
        # Filters
        'blur_light': {'radius': 1.0},
        'blur_medium': {'radius': 2.0},
        'blur_heavy': {'radius': 3.0},
        'motion_blur': {'size': 5},
        'sharpen_light': {'factor': 1.5},
        'sharpen_heavy': {'factor': 3.0},
        
        # Color
        'brighten_20': {'factor': 1.2},
        'darken_20': {'factor': 0.8},
        'high_contrast': {'factor': 1.5},
        'low_contrast': {'factor': 0.7},
        'saturate': {'factor': 1.3},
        'desaturate': {'factor': 0.7},
        
        # Noise
        'noise_light': {'noise_level': 0.05},
        'noise_medium': {'noise_level': 0.1},
        'noise_heavy': {'noise_level': 0.2},
        
        # Compression
        'jpeg_90': {'quality': 90},
        'jpeg_70': {'quality': 70},
        'jpeg_50': {'quality': 50},
        'jpeg_30': {'quality': 30},
    }


def apply_standard_transforms(image: Image.Image) -> Dict[str, TransformResult]:
    """Apply all standard transformations to an image."""
    configs = get_standard_transform_configs()
    results = {}
    
    for name, params in configs.items():
        transform_name = name.split('_')[0]  # Extract base transform name
        if transform_name in ['crop', 'resize', 'rotate', 'blur', 'brighten', 'darken', 'high', 'low', 'saturate', 'desaturate', 'noise', 'jpeg']:
            # Map to actual transform names
            transform_mapping = {
                'crop': 'crop_center',
                'resize': 'resize',
                'rotate': 'rotate',
                'blur': 'gaussian_blur',
                'brighten': 'brightness',
                'darken': 'brightness',
                'high': 'contrast',
                'low': 'contrast',
                'saturate': 'saturation',
                'desaturate': 'saturation',
                'noise': 'gaussian_noise',
                'jpeg': 'jpeg_compression'
            }
            
            actual_transform = transform_mapping.get(transform_name, transform_name)
            result = transform_registry.apply_transform(image, actual_transform, **params)
            results[name] = result
    
    return results