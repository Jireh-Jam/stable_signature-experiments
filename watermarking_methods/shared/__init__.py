"""
Shared utilities for watermarking methods.

This module contains common functions and utilities used across different
watermarking implementations.
"""

from .io import (
    load_image,
    save_image,
    load_images_from_folder,
    save_images_to_folder,
)
from .image_utils import (
    tensor_to_pil,
    pil_to_tensor,
    normalize_image,
    denormalize_image,
    resize_image_tensor,
)
from .logging_utils import setup_logging, get_logger
from .config import load_config, merge_configs

__all__ = [
    # I/O functions
    "load_image",
    "save_image", 
    "load_images_from_folder",
    "save_images_to_folder",
    # Image utilities
    "tensor_to_pil",
    "pil_to_tensor",
    "normalize_image",
    "denormalize_image",
    "resize_image_tensor",
    # Logging
    "setup_logging",
    "get_logger",
    # Configuration
    "load_config",
    "merge_configs",
]