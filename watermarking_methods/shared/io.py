"""
Input/Output utilities for watermarking methods.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Tuple
from PIL import Image
import torch


def load_image(path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Load an image from file.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image or None if loading failed
    """
    try:
        path = Path(path)
        if not path.exists():
            print(f"❌ Image not found: {path}")
            return None
            
        image = Image.open(path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        print(f"❌ Error loading image {path}: {str(e)}")
        return None


def save_image(image: Image.Image, path: Union[str, Path], quality: int = 95) -> bool:
    """
    Save an image to file.
    
    Args:
        image: PIL Image to save
        path: Output file path
        quality: JPEG quality (if saving as JPEG)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension
        ext = path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            image.save(path, 'JPEG', quality=quality, optimize=True)
        elif ext == '.png':
            image.save(path, 'PNG', optimize=True)
        else:
            # Default to PNG for unknown extensions
            image.save(path, 'PNG', optimize=True)
            
        return True
        
    except Exception as e:
        print(f"❌ Error saving image {path}: {str(e)}")
        return False


def load_images_from_folder(
    folder_path: Union[str, Path], 
    extensions: Optional[List[str]] = None,
    max_images: Optional[int] = None
) -> List[Tuple[str, Image.Image]]:
    """
    Load all images from a folder.
    
    Args:
        folder_path: Path to folder containing images
        extensions: List of file extensions to include (default: common image formats)
        max_images: Maximum number of images to load
        
    Returns:
        List of (filename, image) tuples
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    images = []
    count = 0
    
    for file_path in folder_path.iterdir():
        if max_images and count >= max_images:
            break
            
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image = load_image(file_path)
            if image:
                images.append((file_path.name, image))
                count += 1
    
    print(f"📁 Loaded {len(images)} images from {folder_path}")
    return images


def save_images_to_folder(
    images: List[Tuple[str, Image.Image]], 
    output_folder: Union[str, Path],
    prefix: str = "",
    quality: int = 95
) -> int:
    """
    Save multiple images to a folder.
    
    Args:
        images: List of (filename, image) tuples
        output_folder: Output folder path
        prefix: Prefix to add to filenames
        quality: JPEG quality
        
    Returns:
        Number of images successfully saved
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for filename, image in images:
        if prefix:
            filename = f"{prefix}_{filename}"
        output_path = output_folder / filename
        
        if save_image(image, output_path, quality=quality):
            saved_count += 1
    
    print(f"💾 Saved {saved_count}/{len(images)} images to {output_folder}")
    return saved_count