"""
Shared utilities for all watermarking methods.

This module contains common functionality used across different
watermarking implementations, including I/O operations, image utilities,
model utilities, and transformations.
"""

# Re-export key utilities for convenience
from .io import (
    load_image,
    save_image,
    load_images_from_folder,
    get_image_paths,
)

# Lazy imports for utilities that may have heavy dependencies
def _get_image_utils():
    """Lazy import for image_utils to avoid circular imports."""
    from . import image_utils
    return image_utils

def _get_model_utils():
    """Lazy import for model_utils to avoid circular imports."""
    from . import model_utils
    return model_utils

def _get_transforms():
    """Lazy import for transforms to avoid heavy dependencies at import time."""
    from . import transforms
    return transforms

__all__ = [
    # I/O utilities
    "load_image",
    "save_image",
    "load_images_from_folder",
    "get_image_paths",
]
