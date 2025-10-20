"""
Image I/O utilities for watermarking methods.
"""

import os
from pathlib import Path
from typing import List, Union, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def load_image(path: Union[str, Path], mode: str = 'RGB') -> Image.Image:
    """
    Load an image from disk.
    
    Args:
        path: Path to the image file
        mode: PIL image mode (default: 'RGB')
        
    Returns:
        PIL Image object
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = Image.open(path)
    if mode and img.mode != mode:
        img = img.convert(mode)
    return img


def save_image(image: Union[Image.Image, torch.Tensor, np.ndarray], 
               path: Union[str, Path], 
               quality: int = 95,
               **kwargs):
    """
    Save an image to disk.
    
    Args:
        image: Image to save (PIL, tensor, or numpy array)
        path: Output path
        quality: JPEG quality (default: 95)
        **kwargs: Additional arguments for PIL save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to PIL if needed
    if isinstance(image, torch.Tensor):
        image = tensor_to_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Save with appropriate format
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        image.save(path, quality=quality, **kwargs)
    else:
        image.save(path, **kwargs)


def load_batch(paths: List[Union[str, Path]], 
               transform: Optional[transforms.Compose] = None) -> torch.Tensor:
    """
    Load a batch of images.
    
    Args:
        paths: List of image paths
        transform: Optional torchvision transform
        
    Returns:
        Batch tensor of shape (N, C, H, W)
    """
    images = []
    for path in paths:
        img = load_image(path)
        if transform:
            img = transform(img)
        else:
            img = transforms.ToTensor()(img)
        images.append(img)
    
    return torch.stack(images)


def save_batch(images: torch.Tensor, 
               output_dir: Union[str, Path], 
               prefix: str = "img",
               quality: int = 95):
    """
    Save a batch of images.
    
    Args:
        images: Batch tensor of shape (N, C, H, W)
        output_dir: Output directory
        prefix: Filename prefix
        quality: JPEG quality
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        output_path = output_dir / f"{prefix}_{i:04d}.png"
        save_image(img, output_path, quality=quality)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor."""
    return transforms.ToTensor()(image)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
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