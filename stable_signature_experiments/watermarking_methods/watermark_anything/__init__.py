"""
Watermark Anything Method

Implementation of the Watermark Anything technique.
"""

from .method import WatermarkAnythingMethod
from .api import get_backend, embed_image, detect_image, embed_on_path, detect_on_path
from .runner import embed_folder, detect_folder

__all__ = [
    "WatermarkAnythingMethod",
    # API
    "get_backend",
    "embed_image",
    "detect_image",
    "embed_on_path",
    "detect_on_path",
    # Runners
    "embed_folder",
    "detect_folder",
]