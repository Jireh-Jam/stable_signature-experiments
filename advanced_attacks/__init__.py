"""
Advanced watermark attack methods for robustness testing.

This package provides a comprehensive suite of watermark attack methods including:
- Frequency domain attacks
- Diffusion-based attacks  
- Adversarial attacks
- Traditional image processing attacks
- Geometric transformations
"""

from .attacks import WatermarkAttacker
from .attack_registry import AttackRegistry, get_available_attacks
from .frequency_attacks import FrequencyAttacks
from .diffusion_attacks import DiffusionAttacks

__version__ = "1.0.0"
__all__ = [
    "WatermarkAttacker",
    "AttackRegistry", 
    "get_available_attacks",
    "FrequencyAttacks",
    "DiffusionAttacks"
]