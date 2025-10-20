"""
Shared utilities for watermarking methods.

This module contains common functionality used across different watermarking methods:
- Image I/O operations
- Transformation utilities
- Logging helpers
- Common data structures
"""

from .io import load_image, save_image, load_batch, save_batch, image_to_tensor, tensor_to_image
from .transforms import normalize_image, denormalize_image
from .logging_utils import get_logger

__all__ = [
    'load_image',
    'save_image', 
    'load_batch',
    'save_batch',
    'normalize_image',
    'denormalize_image',
    'image_to_tensor',
    'tensor_to_image',
    'get_logger',
]