"""
Utility functions for watermark detection.

This module provides helper functions for message encoding/decoding
and parameter management.
"""

from dataclasses import dataclass
from typing import List, Union
import torch
import numpy as np


def msg2str(msg: Union[np.ndarray, torch.Tensor, List[bool]]) -> str:
    """
    Convert a binary message array to string representation.
    
    Args:
        msg: Binary message as numpy array, tensor, or list of booleans
        
    Returns:
        String of '0's and '1's
    """
    if isinstance(msg, torch.Tensor):
        msg = msg.cpu().numpy()
    elif isinstance(msg, list):
        msg = np.array(msg)
    
    # Handle boolean arrays
    if msg.dtype == bool:
        return "".join(['1' if el else '0' for el in msg.flatten()])
    else:
        # Handle numeric arrays (threshold at 0)
        return "".join(['1' if el > 0 else '0' for el in msg.flatten()])


def str2msg(s: str) -> List[bool]:
    """
    Convert string representation to binary message.
    
    Args:
        s: String of '0's and '1's
        
    Returns:
        List of boolean values
    """
    return [True if el == '1' else False for el in s]


@dataclass
class Params:
    """
    Parameters for HiDDeN encoder/decoder models.
    
    Attributes:
        encoder_depth: Number of blocks in encoder
        encoder_channels: Number of channels in encoder
        decoder_depth: Number of blocks in decoder
        decoder_channels: Number of channels in decoder
        num_bits: Number of bits in watermark message
        attenuation: Type of attenuation ('jnd' or None)
        scale_channels: Whether to scale color channels
        scaling_i: Scaling factor for original image
        scaling_w: Scaling factor for watermark
    """
    encoder_depth: int = 4
    encoder_channels: int = 64
    decoder_depth: int = 8
    decoder_channels: int = 64
    num_bits: int = 48
    attenuation: str = "jnd"
    scale_channels: bool = False
    scaling_i: float = 1.0
    scaling_w: float = 1.5
    
    @classmethod
    def default(cls) -> 'Params':
        """Get default parameters."""
        return cls()
    
    @classmethod
    def from_dict(cls, config: dict) -> 'Params':
        """Create parameters from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})