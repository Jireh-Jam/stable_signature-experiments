"""
Base class for watermarking methods.

This module defines the interface that all watermarking methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np


class BaseWatermarkMethod(ABC):
    """
    Abstract base class for watermarking methods.
    
    All watermarking implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize the watermarking method.
        
        Args:
            name: Human-readable name of the method
        """
        self.name = name
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the watermarking method with required models/parameters.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
        
    @abstractmethod
    def embed_watermark(self, image: Image.Image, message: str) -> Tuple[Image.Image, bool]:
        """
        Embed a watermark message into an image.
        
        Args:
            image: Input PIL Image
            message: Message to embed (e.g., binary string)
            
        Returns:
            Tuple of (watermarked_image, success_flag)
        """
        pass
        
    @abstractmethod
    def detect_watermark(self, image: Image.Image) -> Tuple[bool, float, Optional[str]]:
        """
        Detect watermark in an image.
        
        Args:
            image: Input PIL Image to check
            
        Returns:
            Tuple of (detected, confidence_score, extracted_message)
        """
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this watermarking method.
        
        Returns:
            Dictionary with method information
        """
        return {
            "name": self.name,
            "initialized": self.is_initialized,
            "class": self.__class__.__name__
        }
        
    def __str__(self) -> str:
        return f"{self.name} Watermarking Method"
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self.is_initialized})"