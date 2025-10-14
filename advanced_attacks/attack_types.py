"""
Attack type definitions and configurations.

This module defines the available attack types and their configurations.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any


class AttackType(Enum):
    """Enumeration of available attack types."""
    
    # Traditional attacks
    GAUSSIAN_BLUR = "gaussian_blur"
    GAUSSIAN_NOISE = "gaussian_noise"
    JPEG_COMPRESSION = "jpeg"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    ROTATION = "rotation"
    SCALE = "scale"
    CROP = "crop"
    
    # Advanced denoising
    BM3D = "bm3d"
    
    # Frequency domain attacks
    HIGH_FREQUENCY = "high_frequency"
    
    # Neural compression
    VAE_COMPRESSION = "vae"
    
    # Diffusion-based attacks
    DIFFUSION_INPAINTING = "diffusion_inpainting"
    DIFFUSION_REGENERATION = "diffusion_regeneration"
    DIFFUSION_IMG2IMG = "diffusion_img2img"
    DIFFUSION_RESD = "diffusion_resd"
    
    # Adversarial attacks
    ADVERSARIAL_FGSM = "adversarial_fgsm"
    ADVERSARIAL_PGD = "adversarial_pgd"
    ADVERSARIAL_DEEPFOOL = "adversarial_deepfool"


@dataclass
class AttackConfig:
    """Configuration for an attack."""
    
    attack_type: AttackType
    params: Dict[str, Any]
    description: Optional[str] = None
    
    @classmethod
    def gaussian_blur(cls, kernel_size: int = 5, sigma: float = 1.0) -> 'AttackConfig':
        """Create Gaussian blur attack config."""
        return cls(
            attack_type=AttackType.GAUSSIAN_BLUR,
            params={'kernel_size': kernel_size, 'sigma': sigma},
            description=f"Gaussian blur with kernel={kernel_size}, sigma={sigma}"
        )
    
    @classmethod
    def gaussian_noise(cls, std: float = 0.05) -> 'AttackConfig':
        """Create Gaussian noise attack config."""
        return cls(
            attack_type=AttackType.GAUSSIAN_NOISE,
            params={'std': std},
            description=f"Gaussian noise with std={std}"
        )
    
    @classmethod
    def jpeg_compression(cls, quality: int = 80) -> 'AttackConfig':
        """Create JPEG compression attack config."""
        return cls(
            attack_type=AttackType.JPEG_COMPRESSION,
            params={'quality': quality},
            description=f"JPEG compression with quality={quality}"
        )
    
    @classmethod
    def brightness(cls, factor: float = 1.2) -> 'AttackConfig':
        """Create brightness adjustment attack config."""
        return cls(
            attack_type=AttackType.BRIGHTNESS,
            params={'factor': factor},
            description=f"Brightness adjustment with factor={factor}"
        )
    
    @classmethod
    def contrast(cls, factor: float = 1.2) -> 'AttackConfig':
        """Create contrast adjustment attack config."""
        return cls(
            attack_type=AttackType.CONTRAST,
            params={'factor': factor},
            description=f"Contrast adjustment with factor={factor}"
        )
    
    @classmethod
    def rotation(cls, degrees: float = 15.0) -> 'AttackConfig':
        """Create rotation attack config."""
        return cls(
            attack_type=AttackType.ROTATION,
            params={'degrees': degrees},
            description=f"Rotation by {degrees} degrees"
        )
    
    @classmethod
    def scale(cls, factor: float = 0.5) -> 'AttackConfig':
        """Create scaling attack config."""
        return cls(
            attack_type=AttackType.SCALE,
            params={'factor': factor},
            description=f"Scaling by factor={factor}"
        )
    
    @classmethod
    def crop(cls, ratio: float = 0.5) -> 'AttackConfig':
        """Create cropping attack config."""
        return cls(
            attack_type=AttackType.CROP,
            params={'ratio': ratio},
            description=f"Cropping with ratio={ratio}"
        )
    
    @classmethod
    def high_frequency(cls, threshold: float = 95.0, strength: float = 0.8) -> 'AttackConfig':
        """Create high frequency filtering attack config."""
        return cls(
            attack_type=AttackType.HIGH_FREQUENCY,
            params={'threshold_percentile': threshold, 'filter_strength': strength},
            description=f"High frequency filter with threshold={threshold}%, strength={strength}"
        )
    
    @classmethod
    def diffusion_inpainting(cls, mask_ratio: float = 0.3, prompt: str = "A clean photograph") -> 'AttackConfig':
        """Create diffusion inpainting attack config."""
        return cls(
            attack_type=AttackType.DIFFUSION_INPAINTING,
            params={'mask_ratio': mask_ratio, 'prompt': prompt},
            description=f"Diffusion inpainting with mask_ratio={mask_ratio}"
        )
    
    @classmethod
    def diffusion_regeneration(cls, strength: float = 0.5, prompt: str = "A clean photograph") -> 'AttackConfig':
        """Create diffusion regeneration attack config."""
        return cls(
            attack_type=AttackType.DIFFUSION_REGENERATION,
            params={'strength': strength, 'prompt': prompt},
            description=f"Diffusion regeneration with strength={strength}"
        )
    
    @classmethod
    def adversarial_fgsm(cls, epsilon: float = 0.03) -> 'AttackConfig':
        """Create FGSM adversarial attack config."""
        return cls(
            attack_type=AttackType.ADVERSARIAL_FGSM,
            params={'epsilon': epsilon},
            description=f"FGSM adversarial attack with epsilon={epsilon}"
        )


def get_standard_attack_suite() -> Dict[str, AttackConfig]:
    """
    Get a standard suite of attacks for testing.
    
    Returns:
        Dictionary mapping attack names to configurations
    """
    return {
        # Basic image processing
        'blur_light': AttackConfig.gaussian_blur(kernel_size=3, sigma=0.5),
        'blur_medium': AttackConfig.gaussian_blur(kernel_size=5, sigma=1.0),
        'blur_heavy': AttackConfig.gaussian_blur(kernel_size=7, sigma=2.0),
        
        'noise_light': AttackConfig.gaussian_noise(std=0.01),
        'noise_medium': AttackConfig.gaussian_noise(std=0.05),
        'noise_heavy': AttackConfig.gaussian_noise(std=0.1),
        
        'jpeg_90': AttackConfig.jpeg_compression(quality=90),
        'jpeg_70': AttackConfig.jpeg_compression(quality=70),
        'jpeg_50': AttackConfig.jpeg_compression(quality=50),
        'jpeg_30': AttackConfig.jpeg_compression(quality=30),
        
        # Color and geometry
        'brighten_20': AttackConfig.brightness(factor=1.2),
        'darken_20': AttackConfig.brightness(factor=0.8),
        'high_contrast': AttackConfig.contrast(factor=1.5),
        'low_contrast': AttackConfig.contrast(factor=0.7),
        
        'rotate_5': AttackConfig.rotation(degrees=5.0),
        'rotate_15': AttackConfig.rotation(degrees=15.0),
        'scale_50': AttackConfig.scale(factor=0.5),
        'scale_80': AttackConfig.scale(factor=0.8),
        'crop_70': AttackConfig.crop(ratio=0.7),
        
        # Advanced attacks
        'high_freq_95': AttackConfig.high_frequency(threshold=95.0, strength=0.8),
    }


def get_aggressive_attack_suite() -> Dict[str, AttackConfig]:
    """
    Get an aggressive suite of attacks for stress testing.
    
    Returns:
        Dictionary mapping attack names to configurations
    """
    return {
        'blur_extreme': AttackConfig.gaussian_blur(kernel_size=11, sigma=5.0),
        'noise_extreme': AttackConfig.gaussian_noise(std=0.2),
        'jpeg_10': AttackConfig.jpeg_compression(quality=10),
        'brighten_extreme': AttackConfig.brightness(factor=2.0),
        'darken_extreme': AttackConfig.brightness(factor=0.3),
        'rotate_30': AttackConfig.rotation(degrees=30.0),
        'scale_30': AttackConfig.scale(factor=0.3),
        'crop_50': AttackConfig.crop(ratio=0.5),
        'high_freq_98': AttackConfig.high_frequency(threshold=98.0, strength=0.95),
    }