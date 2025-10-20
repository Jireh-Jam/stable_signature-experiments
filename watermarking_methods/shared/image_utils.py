"""
Image processing utilities for watermarking methods.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def pil_to_tensor(image: Image.Image, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert PIL image to PyTorch tensor.
    
    Args:
        image: PIL Image
        device: Target device for tensor
        
    Returns:
        Tensor with shape (1, C, H, W) and values in [0, 1]
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    if device:
        tensor = tensor.to(device)
    
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert PyTorch tensor to PIL image.
    
    Args:
        tensor: Tensor with shape (1, C, H, W) or (C, H, W) and values in [0, 1]
        
    Returns:
        PIL Image
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Move to CPU and denormalize
    tensor = tensor.cpu().clamp(0, 1)
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def normalize_image(tensor: torch.Tensor, mean: Tuple[float, ...] = (0.5, 0.5, 0.5), 
                   std: Tuple[float, ...] = (0.5, 0.5, 0.5)) -> torch.Tensor:
    """
    Normalize image tensor using mean and std.
    
    Args:
        tensor: Image tensor with values in [0, 1]
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Normalized tensor
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    
    return (tensor - mean) / std


def denormalize_image(tensor: torch.Tensor, mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
                     std: Tuple[float, ...] = (0.5, 0.5, 0.5)) -> torch.Tensor:
    """
    Denormalize image tensor using mean and std.
    
    Args:
        tensor: Normalized image tensor
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    
    return (tensor * std + mean).clamp(0, 1)


def resize_image_tensor(tensor: torch.Tensor, size: Tuple[int, int], 
                       mode: str = 'bilinear') -> torch.Tensor:
    """
    Resize image tensor to specified size.
    
    Args:
        tensor: Image tensor with shape (B, C, H, W)
        size: Target size as (height, width)
        mode: Interpolation mode
        
    Returns:
        Resized tensor
    """
    return torch.nn.functional.interpolate(
        tensor, size=size, mode=mode, align_corners=False
    )


def crop_center(tensor: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
    """
    Center crop image tensor.
    
    Args:
        tensor: Image tensor with shape (B, C, H, W)
        crop_size: Target crop size as (height, width)
        
    Returns:
        Center-cropped tensor
    """
    _, _, h, w = tensor.shape
    crop_h, crop_w = crop_size
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]


def add_gaussian_noise(tensor: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to image tensor.
    
    Args:
        tensor: Image tensor
        noise_level: Standard deviation of noise
        
    Returns:
        Noisy tensor
    """
    noise = torch.randn_like(tensor) * noise_level
    return (tensor + noise).clamp(0, 1)