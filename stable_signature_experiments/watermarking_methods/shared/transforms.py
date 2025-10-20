"""
Common transformation utilities for watermarking methods.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import Union, Tuple, Optional


# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_image(image: torch.Tensor, 
                   mean: Optional[list] = None, 
                   std: Optional[list] = None) -> torch.Tensor:
    """
    Normalize image tensor.
    
    Args:
        image: Input tensor
        mean: Per-channel mean (default: ImageNet)
        std: Per-channel std (default: ImageNet)
        
    Returns:
        Normalized tensor
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    normalize = transforms.Normalize(mean=mean, std=std)
    
    if image.dim() == 3:
        return normalize(image)
    elif image.dim() == 4:
        return torch.stack([normalize(img) for img in image])
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")


def denormalize_image(image: torch.Tensor,
                     mean: Optional[list] = None,
                     std: Optional[list] = None) -> torch.Tensor:
    """
    Denormalize image tensor.
    
    Args:
        image: Normalized input tensor
        mean: Per-channel mean (default: ImageNet)
        std: Per-channel std (default: ImageNet)
        
    Returns:
        Denormalized tensor
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    
    if image.device != mean.device:
        mean = mean.to(image.device)
        std = std.to(image.device)
    
    return image * std + mean


def resize_image(image: torch.Tensor, 
                size: Union[int, Tuple[int, int]], 
                mode: str = 'bilinear') -> torch.Tensor:
    """
    Resize image tensor.
    
    Args:
        image: Input tensor
        size: Target size (int or (h, w))
        mode: Interpolation mode
        
    Returns:
        Resized tensor
    """
    if isinstance(size, int):
        size = (size, size)
    
    return F.interpolate(image, size=size, mode=mode, align_corners=False)


def center_crop(image: torch.Tensor, size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """
    Center crop image tensor.
    
    Args:
        image: Input tensor
        size: Crop size (int or (h, w))
        
    Returns:
        Cropped tensor
    """
    if isinstance(size, int):
        size = (size, size)
    
    _, _, h, w = image.shape
    th, tw = size
    
    if h < th or w < tw:
        raise ValueError(f"Image size ({h}, {w}) smaller than crop size ({th}, {tw})")
    
    x1 = (w - tw) // 2
    y1 = (h - th) // 2
    
    return image[:, :, y1:y1+th, x1:x1+tw]


def random_crop(image: torch.Tensor, size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """
    Random crop image tensor.
    
    Args:
        image: Input tensor
        size: Crop size (int or (h, w))
        
    Returns:
        Cropped tensor
    """
    if isinstance(size, int):
        size = (size, size)
    
    _, _, h, w = image.shape
    th, tw = size
    
    if h < th or w < tw:
        raise ValueError(f"Image size ({h}, {w}) smaller than crop size ({th}, {tw})")
    
    x1 = torch.randint(0, w - tw + 1, (1,)).item()
    y1 = torch.randint(0, h - th + 1, (1,)).item()
    
    return image[:, :, y1:y1+th, x1:x1+tw]


def add_gaussian_noise(image: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """
    Add Gaussian noise to image.
    
    Args:
        image: Input tensor
        std: Noise standard deviation
        
    Returns:
        Noisy image
    """
    noise = torch.randn_like(image) * std
    return torch.clamp(image + noise, 0, 1)


def jpeg_compress(image: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Simulate JPEG compression.
    
    Args:
        image: Input tensor
        quality: JPEG quality (1-100)
        
    Returns:
        Compressed image tensor
    """
    # This is a placeholder - actual JPEG compression requires PIL conversion
    # In practice, this would convert to PIL, compress, and convert back
    # For now, we'll add some quantization noise to simulate compression
    
    # Simulate quantization
    levels = quality / 100.0 * 255
    quantized = torch.round(image * levels) / levels
    
    # Add some blocking artifacts
    block_size = max(1, int(8 * (1 - quality / 100.0)))
    if block_size > 1:
        _, _, h, w = image.shape
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = image[:, :, i:i+block_size, j:j+block_size]
                mean = block.mean(dim=(2, 3), keepdim=True)
                quantized[:, :, i:i+block_size, j:j+block_size] = mean
    
    return quantized


def image_to_tensor(image) -> torch.Tensor:
    """
    Convert PIL Image to tensor.
    
    Args:
        image: PIL Image
        
    Returns:
        Tensor of shape (C, H, W)
    """
    return transforms.ToTensor()(image)


def tensor_to_image(tensor: torch.Tensor):
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Input tensor
        
    Returns:
        PIL Image
    """
    from PIL import Image
    
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    
    # Denormalize if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Convert to numpy and then PIL
    array = (tensor * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
    return Image.fromarray(array)