"""
Watermark detection and analysis tools.

This package provides tools for:
- Detecting watermarks in images
- Analyzing watermark robustness
- Evaluating detection accuracy
- Model management and inference
"""

from .detector import WatermarkDetector
from .models import ModelManager, load_detection_model
from .evaluation import DetectionEvaluator

__version__ = "1.0.0"
__all__ = [
    "WatermarkDetector",
    "ModelManager", 
    "load_detection_model",
    "DetectionEvaluator"
]