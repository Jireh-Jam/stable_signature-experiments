"""
Image transformation utilities for watermark robustness testing.

This module provides a comprehensive set of image transformations
commonly used to test watermark robustness.
"""

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from typing import Dict, Callable, Tuple, List
import random
from common.transforms_registry import registry
from common.logging_utils import get_logger

logger = get_logger(__name__)


class ImageTransformations:
    """
    Collection of image transformations for watermark robustness testing.
    
    Each transformation is designed to simulate real-world image modifications
    that might occur during image sharing, compression, or editing.
    """
    
    @staticmethod
    def crop_center(image: Image.Image, crop_percentage: float = 0.1) -> Image.Image:
        """
        Crop the image from the center, removing edges.
        
        Args:
            image: Input PIL Image
            crop_percentage: Percentage of image to crop from each edge (0.0 to 0.5)
            
        Returns:
            Cropped PIL Image
        """
        width, height = image.size
        crop_w = int(width * crop_percentage)
        crop_h = int(height * crop_percentage)
        
        left = crop_w
        top = crop_h
        right = width - crop_w
        bottom = height - crop_h
        
        return image.crop((left, top, right, bottom))
    
    @staticmethod
    def crop_random(image: Image.Image, crop_percentage: float = 0.1) -> Image.Image:
        """
        Crop the image randomly, removing a portion from a random location.
        
        Args:
            image: Input PIL Image
            crop_percentage: Percentage of image to crop
            
        Returns:
            Cropped PIL Image
        """
        width, height = image.size
        crop_w = int(width * crop_percentage)
        crop_h = int(height * crop_percentage)
        
        # Random starting position
        left = random.randint(0, crop_w)
        top = random.randint(0, crop_h)
        right = width - random.randint(0, crop_w)
        bottom = height - random.randint(0, crop_h)
        
        return image.crop((left, top, right, bottom))
    
    @staticmethod
    def blur_gaussian(image: Image.Image, radius: float = 1.0) -> Image.Image:
        """
        Apply Gaussian blur to the image.
        
        Args:
            image: Input PIL Image
            radius: Blur radius (higher = more blur)
            
        Returns:
            Blurred PIL Image
        """
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    @staticmethod
    def blur_motion(image: Image.Image, size: int = 5) -> Image.Image:
        """
        Apply motion blur to simulate camera movement.
        
        Args:
            image: Input PIL Image
            size: Size of motion blur kernel
            
        Returns:
            Motion-blurred PIL Image
        """
        # Create motion blur kernel
        kernel = ImageFilter.Kernel((size, size), 
                                   [1/size] * size + [0] * (size * size - size))
        return image.filter(kernel)
    
    @staticmethod
    def adjust_brightness(image: Image.Image, factor: float = 1.2) -> Image.Image:
        """
        Adjust image brightness.
        
        Args:
            image: Input PIL Image
            factor: Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
            
        Returns:
            Brightness-adjusted PIL Image
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def adjust_contrast(image: Image.Image, factor: float = 1.2) -> Image.Image:
        """
        Adjust image contrast.
        
        Args:
            image: Input PIL Image
            factor: Contrast factor (1.0 = no change, >1.0 = more contrast)
            
        Returns:
            Contrast-adjusted PIL Image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def adjust_saturation(image: Image.Image, factor: float = 1.2) -> Image.Image:
        """
        Adjust image colour saturation.
        
        Args:
            image: Input PIL Image
            factor: Saturation factor (1.0 = no change, 0.0 = grayscale)
            
        Returns:
            Saturation-adjusted PIL Image
        """
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def resize_image(image: Image.Image, scale: float = 0.8) -> Image.Image:
        """
        Resize the image by a scale factor.
        
        Args:
            image: Input PIL Image
            scale: Scale factor (1.0 = no change, 0.5 = half size, 2.0 = double size)
            
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def rotate_image(image: Image.Image, angle: float = 5.0) -> Image.Image:
        """
        Rotate the image by a specified angle.
        
        Args:
            image: Input PIL Image
            angle: Rotation angle in degrees (positive = clockwise)
            
        Returns:
            Rotated PIL Image
        """
        return image.rotate(angle, expand=True, fillcolor='white')
    
    @staticmethod
    def add_noise(image: Image.Image, noise_level: float = 0.1) -> Image.Image:
        """
        Add random noise to the image.
        
        Args:
            image: Input PIL Image
            noise_level: Noise intensity (0.0 to 1.0)
            
        Returns:
            Noisy PIL Image
        """
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        
        # Generate noise
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        
        # Add noise and clip values
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(noisy_array)
    
    @staticmethod
    def jpeg_compression(image: Image.Image, quality: int = 85) -> Image.Image:
        """
        Simulate JPEG compression by saving and reloading the image.
        
        Args:
            image: Input PIL Image
            quality: JPEG quality (1-100, higher = better quality)
            
        Returns:
            JPEG-compressed PIL Image
        """
        import io
        
        # Save to bytes with JPEG compression
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load from bytes
        return Image.open(buffer)
    
    @classmethod
    def get_standard_transformations(cls) -> Dict[str, Callable[[Image.Image], Image.Image]]:
        """
        Get a dictionary of standard transformations for testing.
        
        Returns:
            Dictionary mapping transformation names to functions
        """
        return {
            # Cropping transformations
            'crop_10_percent': lambda img: cls.crop_center(img, 0.1),
            'crop_20_percent': lambda img: cls.crop_center(img, 0.2),
            'crop_random_15': lambda img: cls.crop_random(img, 0.15),
            
            # Blur transformations
            'blur_light': lambda img: cls.blur_gaussian(img, 1.0),
            'blur_medium': lambda img: cls.blur_gaussian(img, 2.0),
            'blur_heavy': lambda img: cls.blur_gaussian(img, 3.0),
            'motion_blur': lambda img: cls.blur_motion(img, 5),
            
            # Brightness and contrast
            'brighten_20': lambda img: cls.adjust_brightness(img, 1.2),
            'darken_20': lambda img: cls.adjust_brightness(img, 0.8),
            'brighten_50': lambda img: cls.adjust_brightness(img, 1.5),
            'darken_50': lambda img: cls.adjust_brightness(img, 0.5),
            'high_contrast': lambda img: cls.adjust_contrast(img, 1.5),
            'low_contrast': lambda img: cls.adjust_contrast(img, 0.7),
            
            # Colour adjustments
            'saturate': lambda img: cls.adjust_saturation(img, 1.3),
            'desaturate': lambda img: cls.adjust_saturation(img, 0.7),
            'grayscale': lambda img: cls.adjust_saturation(img, 0.0),
            
            # Geometric transformations
            'resize_80': lambda img: cls.resize_image(img, 0.8),
            'resize_60': lambda img: cls.resize_image(img, 0.6),
            'resize_120': lambda img: cls.resize_image(img, 1.2),
            'rotate_5': lambda img: cls.rotate_image(img, 5.0),
            'rotate_neg5': lambda img: cls.rotate_image(img, -5.0),
            'rotate_10': lambda img: cls.rotate_image(img, 10.0),
            
            # Noise and compression
            'noise_light': lambda img: cls.add_noise(img, 0.05),
            'noise_medium': lambda img: cls.add_noise(img, 0.1),
            'jpeg_90': lambda img: cls.jpeg_compression(img, 90),
            'jpeg_70': lambda img: cls.jpeg_compression(img, 70),
            'jpeg_50': lambda img: cls.jpeg_compression(img, 50),
        }
    
    @classmethod
    def get_aggressive_transformations(cls) -> Dict[str, Callable[[Image.Image], Image.Image]]:
        """
        Get a dictionary of more aggressive transformations for stress testing.
        
        Returns:
            Dictionary mapping transformation names to functions
        """
        return {
            'crop_30_percent': lambda img: cls.crop_center(img, 0.3),
            'blur_extreme': lambda img: cls.blur_gaussian(img, 5.0),
            'brighten_extreme': lambda img: cls.adjust_brightness(img, 2.0),
            'darken_extreme': lambda img: cls.adjust_brightness(img, 0.3),
            'resize_40': lambda img: cls.resize_image(img, 0.4),
            'rotate_30': lambda img: cls.rotate_image(img, 30.0),
            'noise_heavy': lambda img: cls.add_noise(img, 0.2),
            'jpeg_30': lambda img: cls.jpeg_compression(img, 30),
            'jpeg_10': lambda img: cls.jpeg_compression(img, 10),
        }
    
    @classmethod
    def apply_transformation_chain(cls, image: Image.Image,
                                   transformations: List[Tuple[str, Callable]]) -> Image.Image:
        """
        Apply a chain of transformations to an image.
        
        Args:
            image: Input PIL Image
            transformations: List of (name, function) tuples
            
        Returns:
            Transformed PIL Image
        """
        result = image.copy()
        
        for name, transform_func in transformations:
            try:
                result = transform_func(result)
                logger.debug(f"Applied transform: {name}")
            except Exception as e:
                logger.warning(f"Error applying transform '{name}': {e}")
                
        return result


# Register standard transforms into the shared registry for discoverability
_std = ImageTransformations.get_standard_transformations()
for _name, _fn in _std.items():
    try:
        registry.register(_name, _fn, overwrite=False)
    except Exception:
        # Avoid raising if already registered
        pass

_agg = ImageTransformations.get_aggressive_transformations()
for _name, _fn in _agg.items():
    try:
        registry.register(_name, _fn, overwrite=False)
    except Exception:
        pass