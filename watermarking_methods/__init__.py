"""
Watermarking Methods Package

This package contains implementations for various watermarking techniques:
- Stable Signature
- TrustMark  
- Watermark Anything

Each method provides a consistent interface for embedding and detecting watermarks.
"""

from typing import Optional
from .base import BaseWatermarkMethod

__version__ = "1.0.0"
__author__ = "Watermarking Research Team"

# Available watermarking methods
AVAILABLE_METHODS = [
    "stable_signature",
    "trustmark",
    "watermark_anything",
]

def get_method(method_name: str) -> BaseWatermarkMethod:
    """
    Factory function to get a watermarking method instance.
    
    Args:
        method_name: Name of the watermarking method
        
    Returns:
        Instance of the requested watermarking method
        
    Raises:
        ValueError: If method_name is not supported
    """
    method_name = method_name.lower().strip().replace(" ", "_")

    # Accept common synonyms for user friendliness
    synonyms = {
        "stable_signature": {"stable_signature", "stable-signature", "stable signature", "stable", "ss"},
        "trustmark": {"trustmark", "trust_mark", "trust-mark", "tm"},
        "watermark_anything": {
            "watermark_anything",
            "watermark-anything",
            "watermark anything",
            "wam",
            "watermark_any",
        },
    }

    def match(name: str) -> Optional[str]:
        for canon, alts in synonyms.items():
            if name in alts:
                return canon
        return None

    method_name = match(method_name) or method_name
    
    if method_name == "stable_signature":
        from .stable_signature import StableSignatureMethod
        return StableSignatureMethod()
    elif method_name == "trustmark":
        from .trustmark import TrustMarkMethod
        return TrustMarkMethod()
    elif method_name == "watermark_anything":
        from .watermark_anything import WatermarkAnythingMethod
        return WatermarkAnythingMethod()
    else:
        raise ValueError(f"Unknown watermarking method: {method_name}. "
                        f"Available methods: {AVAILABLE_METHODS}")