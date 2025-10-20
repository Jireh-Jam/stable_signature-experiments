"""
I/O utilities for loading and saving images and data.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
import functools


@functools.lru_cache()
def get_image_paths(directory: Union[str, Path]) -> List[str]:
    """
    Recursively get all image paths from a directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        Sorted list of image file paths
    """
    directory = Path(directory)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if Path(filename).suffix.lower() in image_extensions:
                paths.append(os.path.join(root, filename))
    
    return sorted(paths)


def load_image(path: Union[str, Path], mode: str = 'RGB') -> Image.Image:
    """
    Load an image from a file path.
    
    Args:
        path: Path to image file
        mode: PIL image mode (default: 'RGB')
        
    Returns:
        PIL Image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file cannot be loaded as an image
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    try:
        image = Image.open(path)
        if mode and image.mode != mode:
            image = image.convert(mode)
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from {path}: {str(e)}")


def save_image(
    image: Image.Image,
    path: Union[str, Path],
    create_dirs: bool = True,
    quality: int = 95,
    **kwargs
) -> None:
    """
    Save an image to a file path.
    
    Args:
        image: PIL Image to save
        path: Destination file path
        create_dirs: Whether to create parent directories if they don't exist
        quality: JPEG quality (1-100, default: 95)
        **kwargs: Additional arguments passed to Image.save()
    """
    path = Path(path)
    
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set quality for JPEG images
    save_kwargs = kwargs.copy()
    if path.suffix.lower() in {'.jpg', '.jpeg'}:
        save_kwargs.setdefault('quality', quality)
        save_kwargs.setdefault('optimize', True)
    
    image.save(path, **save_kwargs)


def load_images_from_folder(
    folder: Union[str, Path],
    max_images: Optional[int] = None,
    mode: str = 'RGB'
) -> List[Image.Image]:
    """
    Load all images from a folder.
    
    Args:
        folder: Path to folder containing images
        max_images: Maximum number of images to load (None = all)
        mode: PIL image mode (default: 'RGB')
        
    Returns:
        List of PIL Images
    """
    image_paths = get_image_paths(folder)
    
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    images = []
    for path in image_paths:
        try:
            image = load_image(path, mode=mode)
            images.append(image)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {str(e)}")
    
    return images
