"""Basic image I/O helpers used across methods."""
from __future__ import annotations

from pathlib import Path
from typing import Union
from PIL import Image


PathLike = Union[str, Path]


def load_image(path: PathLike) -> Image.Image:
    p = Path(path)
    return Image.open(p)


def save_image(image: Image.Image, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    image.save(p)
