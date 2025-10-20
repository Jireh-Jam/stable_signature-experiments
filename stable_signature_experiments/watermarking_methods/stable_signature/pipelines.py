"""
Pipeline functions for Stable Signature watermarking.
"""

import os
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..shared.io import load_image, save_image, load_batch, save_batch
from ..shared.transforms import normalize_image, denormalize_image
from .core.models import load_decoder, get_watermark_embedder


def run_watermark(
    input_images: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    message: str = None,
    decoder_path: str = None,
    device: str = None,
    batch_size: int = 1,
    **kwargs
) -> Dict[str, any]:
    """
    Run watermarking pipeline on images.
    
    Args:
        input_images: Input image path(s)
        output_dir: Output directory
        message: Watermark message (default: random)
        decoder_path: Path to decoder model
        device: Device to use (default: auto)
        batch_size: Batch size for processing
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with results
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Handle single image vs directory
    if isinstance(input_images, (str, Path)):
        input_path = Path(input_images)
        if input_path.is_dir():
            image_paths = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
        else:
            image_paths = [input_path]
    else:
        image_paths = [Path(p) for p in input_images]
    
    # Load models
    decoder = load_decoder(decoder_path, device=device)
    embedder = get_watermark_embedder(decoder, device=device)
    
    # Generate message if not provided
    if message is None:
        message = generate_random_message(48)  # 48-bit default
    
    # Process images
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_results = embed_watermark_batch(
            batch_paths, embedder, message, output_dir, device=device
        )
        results.extend(batch_results)
    
    return {
        'num_processed': len(results),
        'output_dir': str(output_dir),
        'message': message,
        'results': results
    }


def embed_watermark(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    message: str,
    decoder_path: str = None,
    device: str = None,
    **kwargs
) -> Union[Image.Image, torch.Tensor]:
    """
    Embed watermark in a single image.
    
    Args:
        image: Input image
        message: Watermark message
        decoder_path: Path to decoder model
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Watermarked image (same type as input)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    decoder = load_decoder(decoder_path, device=device)
    embedder = get_watermark_embedder(decoder, device=device)
    
    # Convert to tensor if needed
    return_pil = isinstance(image, Image.Image)
    return_numpy = isinstance(image, np.ndarray)
    
    if return_pil:
        image = transforms.ToTensor()(image)
    elif return_numpy:
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    
    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Embed watermark
    watermarked = embedder(image.to(device), message)
    
    # Convert back to original format
    if return_pil:
        return tensor_to_image(watermarked[0])
    elif return_numpy:
        return (watermarked[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    else:
        return watermarked


def detect_watermark(
    image: Union[Image.Image, torch.Tensor, np.ndarray, str, Path],
    decoder_path: str = None,
    device: str = None,
    return_confidence: bool = True,
    **kwargs
) -> Union[str, Tuple[str, float]]:
    """
    Detect watermark in an image.
    
    Args:
        image: Input image or path
        decoder_path: Path to decoder model
        device: Device to use
        return_confidence: Whether to return confidence score
        **kwargs: Additional arguments
        
    Returns:
        Detected message (and confidence if requested)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    decoder = load_decoder(decoder_path, device=device)
    
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = load_image(image)
    
    # Convert to tensor
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
    elif isinstance(image, np.ndarray):
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    
    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Detect watermark
    with torch.no_grad():
        decoded = decoder(image.to(device))
        
    # Convert to message string
    message = bits_to_string(decoded[0])
    
    if return_confidence:
        confidence = calculate_confidence(decoded[0])
        return message, confidence
    else:
        return message


# Helper functions
def embed_watermark_batch(
    image_paths: List[Path],
    embedder,
    message: str,
    output_dir: Path,
    device: str = 'cuda'
) -> List[Dict]:
    """Embed watermark in a batch of images."""
    results = []
    
    # Load images
    images = load_batch(image_paths)
    images = images.to(device)
    
    # Embed watermarks
    watermarked = embedder(images, message)
    
    # Save results
    for i, (path, wm_img) in enumerate(zip(image_paths, watermarked)):
        output_path = output_dir / f"watermarked_{path.name}"
        save_image(wm_img, output_path)
        
        results.append({
            'input': str(path),
            'output': str(output_path),
            'success': True
        })
    
    return results


def generate_random_message(num_bits: int = 48) -> str:
    """Generate a random binary message."""
    return ''.join(np.random.choice(['0', '1'], size=num_bits))


def bits_to_string(bits: torch.Tensor) -> str:
    """Convert bit tensor to string."""
    binary = (bits > 0.5).float().cpu().numpy()
    return ''.join(str(int(b)) for b in binary)


def calculate_confidence(bits: torch.Tensor) -> float:
    """Calculate confidence score for detected bits."""
    # Use distance from 0.5 as confidence measure
    confidence = torch.abs(bits - 0.5).mean().item() * 2
    return min(1.0, confidence)


# Import torchvision locally to avoid circular imports
from torchvision import transforms
from ..shared.transforms import normalize_image, denormalize_image
from ..shared.io import tensor_to_image