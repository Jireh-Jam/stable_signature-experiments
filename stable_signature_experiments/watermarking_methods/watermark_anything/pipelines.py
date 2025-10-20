"""
Notebook-friendly pipelines for Watermark Anything (WAM).
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List

from .runner import embed_folder as _embed_folder, detect_folder as _detect_folder


def generate_images(
    input_dir: str,
    output_dir: str,
    message: str,
    max_images: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Embed a message into all images in a folder.

    Returns a list of per-file result dicts.
    """
    return _embed_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        message=message,
        max_images=max_images,
        config=config,
    )


def detect_images(
    dir_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Detect watermarks on all images in a folder."""
    return _detect_folder(dir_path=dir_path, config=config)
