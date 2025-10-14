"""
Image processing utilities for watermark attacks and detection.

This module provides common image I/O, preprocessing, and utility functions
used across the adversarial ML tooling pipeline.
"""

import logging
from typing import Union, Tuple, Optional, List
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio

logger = logging.getLogger(__name__)

# Standard image normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard transforms
NORMALIZE_IMAGENET = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
UNNORMALIZE_IMAGENET = transforms.Normalize(
    mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1/s for s in IMAGENET_STD]
)
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE_IMAGENET
])


def load_image(image_path: Union[str, Path], 
               target_size: Optional[Tuple[int, int]] = None,
               mode: str = 'RGB') -> Image.Image:
    """
    Load an image from file with optional resizing.
    
    Args:
        image_path: Path to the image file
        target_size: Optional (width, height) to resize to
        mode: PIL image mode ('RGB', 'L', etc.)
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    try:
        image = Image.open(image_path).convert(mode)
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {str(e)}")


def save_image(image: Image.Image, 
               output_path: Union[str, Path],
               quality: int = 95) -> None:
    """
    Save a PIL image to file.
    
    Args:
        image: PIL Image to save
        output_path: Output file path
        quality: JPEG quality (if saving as JPEG)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            image.save(output_path, 'JPEG', quality=quality)
        else:
            image.save(output_path)
        logger.debug(f"Saved image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {str(e)}")
        raise


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV BGR image to PIL RGB image.
    
    Args:
        cv2_image: OpenCV image in BGR format
        
    Returns:
        PIL Image in RGB format
    """
    if len(cv2_image.shape) == 3:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2_image
    return Image.fromarray(rgb_image)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL RGB image to OpenCV BGR image.
    
    Args:
        pil_image: PIL Image in RGB format
        
    Returns:
        OpenCV image in BGR format
    """
    rgb_array = np.array(pil_image)
    if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return rgb_array


def tensor_to_pil(tensor: torch.Tensor, 
                  unnormalize: bool = True) -> Image.Image:
    """
    Convert tensor to PIL image.
    
    Args:
        tensor: Input tensor (C, H, W) or (1, C, H, W)
        unnormalize: Whether to apply ImageNet unnormalization
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if unnormalize:
        tensor = UNNORMALIZE_IMAGENET(tensor)
    
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)


def pil_to_tensor(image: Image.Image, 
                  normalize: bool = True,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert PIL image to tensor.
    
    Args:
        image: PIL Image
        normalize: Whether to apply ImageNet normalization
        device: Target device for tensor
        
    Returns:
        Tensor (C, H, W)
    """
    tensor = transforms.ToTensor()(image)
    
    if normalize:
        tensor = NORMALIZE_IMAGENET(tensor)
    
    if device:
        tensor = tensor.to(device)
    
    return tensor


def calculate_image_metrics(img1: Union[np.ndarray, Image.Image], 
                           img2: Union[np.ndarray, Image.Image]) -> dict:
    """
    Calculate image quality metrics between two images.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        Dictionary with PSNR and SSIM values
    """
    # Convert to numpy arrays if needed
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    
    # Ensure same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")
    
    # Calculate PSNR
    psnr_value = peak_signal_noise_ratio(img1, img2)
    
    # Calculate SSIM
    if len(img1.shape) == 3:
        # For color images, convert to grayscale for SSIM
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.shape[2] == 3 else img1[:, :, 0]
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.shape[2] == 3 else img2[:, :, 0]
        ssim_value = ssim(img1_gray, img2_gray)
    else:
        ssim_value = ssim(img1, img2)
    
    return {
        'psnr': float(psnr_value),
        'ssim': float(ssim_value)
    }


def ensure_image_size(image: Union[Image.Image, np.ndarray], 
                      target_size: Tuple[int, int],
                      method: str = 'resize') -> Union[Image.Image, np.ndarray]:
    """
    Ensure image has target size through resize or crop.
    
    Args:
        image: Input image
        target_size: (width, height) target size
        method: 'resize', 'crop_center', or 'pad'
        
    Returns:
        Image with target size
    """
    if isinstance(image, np.ndarray):
        current_h, current_w = image.shape[:2]
        target_w, target_h = target_size
        
        if method == 'resize':
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        elif method == 'crop_center':
            start_x = max(0, (current_w - target_w) // 2)
            start_y = max(0, (current_h - target_h) // 2)
            return image[start_y:start_y+target_h, start_x:start_x+target_w]
    else:  # PIL Image
        if method == 'resize':
            return image.resize(target_size, Image.Resampling.LANCZOS)
        elif method == 'crop_center':
            current_w, current_h = image.size
            target_w, target_h = target_size
            left = (current_w - target_w) // 2
            top = (current_h - target_h) // 2
            return image.crop((left, top, left + target_w, top + target_h))
    
    return image


def validate_image_path(path: Union[str, Path]) -> Path:
    """
    Validate that a path points to a valid image file.
    
    Args:
        path: Path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or not an image
    """
    path = Path(path)
    
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Invalid image extension: {path.suffix}")
    
    return path


def get_image_paths(directory: Union[str, Path], 
                   recursive: bool = True) -> List[Path]:
    """
    Get all image paths from a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    image_paths = []
    for path in directory.glob(pattern):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            image_paths.append(path)
    
    return sorted(image_paths)


def create_image_difference(img1: Image.Image, 
                          img2: Image.Image,
                          amplify: float = 10.0) -> Image.Image:
    """
    Create a visual difference image between two images.
    
    Args:
        img1: First image
        img2: Second image
        amplify: Factor to amplify differences for visibility
        
    Returns:
        Difference image
    """
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    
    diff = np.abs(arr1 - arr2) * amplify
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    return Image.fromarray(diff)