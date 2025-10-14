"""
Advanced watermark attacks module.

This module provides various attack methods to test watermark robustness,
including traditional image processing attacks and advanced AI-based attacks.
"""

from .attacks import WatermarkAttacker
from .attack_types import AttackType, AttackConfig

__version__ = "0.1.0"
__all__ = ['WatermarkAttacker', 'AttackType', 'AttackConfig']