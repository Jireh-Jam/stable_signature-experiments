from pathlib import Path
from typing import Tuple, Optional, Union

from PIL import Image


def load_image_rgb(path: Union[str, Path], size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Load an image as RGB, optionally resizing.

    Args:
        path: Path to image file.
        size: Optional (width, height) to resize to.

    Returns:
        PIL Image in RGB mode.
    """
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img


def save_image(image: Image.Image, path: Union[str, Path]) -> None:
    """Save a PIL image to disk creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    image.save(p)
