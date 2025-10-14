"""
Watermark detection module.

This module provides functionality for detecting watermarks in images
using HiDDeN (Hidden Deep Neural Networks) models.
"""

from .detector import WatermarkDetector
from .models import HiddenDecoder, HiddenEncoder, EncoderWithJND
from .utils import msg2str, str2msg, Params

__version__ = "0.1.0"
__all__ = [
    'WatermarkDetector',
    'HiddenDecoder',
    'HiddenEncoder',
    'EncoderWithJND',
    'msg2str',
    'str2msg',
    'Params'
]