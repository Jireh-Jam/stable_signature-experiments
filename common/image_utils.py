"""
Common image processing utilities for watermark detection and attack operations.

This module provides shared image processing functions used across the codebase.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Union, Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Create standard transforms
NORMALIZE_IMAGENET = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
UNNORMALIZE_IMAGENET = transforms.Normalize(
    mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1/s for s in IMAGENET_STD]
)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE_IMAGENET
])


def load_image(
    path: str,
    size: Optional[Tuple[int, int]] = None,
    mode: str = 'RGB'
) -> Image.Image:
    """
    Load an image from file path with optional resizing.
    
    Args:
        path: Path to the image file
        size: Optional target size as (width, height) tuple
        mode: PIL image mode (default: 'RGB')
        
    Returns:
        PIL Image object
        
    Raises:
        IOError: If image cannot be loaded
    """
    try:
        img = Image.open(path).convert(mode)
        if size is not None:
            img = img.resize(size, Image.Resampling.BICUBIC)
        logger.debug(f"Loaded image from {path}, size: {img.size}")
        return img
    except Exception as e:
        logger.error(f"Failed to load image from {path}: {str(e)}")
        raise IOError(f"Cannot load image from {path}: {str(e)}")


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format (BGR).
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV image array in BGR format
    """
    # Convert PIL to numpy array
    img_array = np.array(pil_image)
    
    # Convert RGB to BGR if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array


def cv2_to_pil(cv2_image: np.ndarray, mode: str = 'RGB') -> Image.Image:
    """
    Convert OpenCV image to PIL format.
    
    Args:
        cv2_image: OpenCV image array (BGR)
        mode: Target PIL mode (default: 'RGB')
        
    Returns:
        PIL Image object
    """
    # Convert BGR to RGB if needed
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
        img_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    else:
        img_array = cv2_image
    
    return Image.fromarray(img_array).convert(mode)


def tensor_to_image(
    tensor: torch.Tensor,
    unnormalize: bool = True
) -> Union[Image.Image, np.ndarray]:
    """
    Convert a tensor to PIL Image.
    
    Args:
        tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
        unnormalize: Whether to apply ImageNet unnormalization
        
    Returns:
        PIL Image if single image, numpy array if batch
    """
    # Remove batch dimension if present
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    # Move to CPU and clone
    tensor = tensor.cpu().clone()
    
    # Unnormalize if requested
    if unnormalize:
        tensor = UNNORMALIZE_IMAGENET(tensor)
    
    # Clip to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    if tensor.dim() == 3:
        # Single image
        return transforms.ToPILImage()(tensor)
    else:
        # Batch of images
        return tensor.numpy()


def image_to_tensor(
    image: Union[Image.Image, np.ndarray],
    normalize: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert PIL Image or numpy array to normalized tensor.
    
    Args:
        image: Input image (PIL or numpy)
        normalize: Whether to apply ImageNet normalization
        device: Target device for tensor
        
    Returns:
        Normalized tensor of shape (C, H, W)
    """
    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to tensor
    tensor = transforms.ToTensor()(image)
    
    # Normalize if requested
    if normalize:
        tensor = NORMALIZE_IMAGENET(tensor)
    
    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def ensure_dimensions(
    image: np.ndarray,
    target_h: int,
    target_w: int
) -> np.ndarray:
    """
    Ensure image has specific dimensions by resizing if necessary.
    
    Args:
        image: Input image array
        target_h: Target height
        target_w: Target width
        
    Returns:
        Resized image array
    """
    current_h, current_w = image.shape[:2]
    
    if current_h != target_h or current_w != target_w:
        logger.debug(f"Resizing image from {current_w}x{current_h} to {target_w}x{target_h}")
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return image


def prepare_image_for_model(
    image_path: str,
    size: Tuple[int, int] = (512, 512),
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load and prepare an image for model input.
    
    Args:
        image_path: Path to the image
        size: Target size (width, height)
        device: Target device for tensor
        
    Returns:
        Tuple of (normalized tensor, original PIL image)
    """
    # Load image
    img = load_image(image_path, size=size)
    
    # Convert to tensor
    img_tensor = image_to_tensor(img, normalize=True, device=device)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, img


def save_image_safely(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    path: str,
    quality: int = 95
) -> bool:
    """
    Save an image with error handling.
    
    Args:
        image: Image to save (PIL, numpy, or tensor)
        path: Output path
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert to PIL if needed
        if isinstance(image, torch.Tensor):
            image = tensor_to_image(image)
        elif isinstance(image, np.ndarray):
            image = cv2_to_pil(image)
        
        # Save based on extension
        if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
            image.save(path, 'JPEG', quality=quality)
        else:
            image.save(path)
        
        logger.info(f"Saved image to {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save image to {path}: {str(e)}")
        return False