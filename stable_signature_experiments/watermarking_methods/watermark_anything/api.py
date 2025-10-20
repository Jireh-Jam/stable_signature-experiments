"""
Public API for Watermark Anything (WAM).

Simple, import-friendly functions for notebooks and pipelines.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
from PIL import Image

from .backend import WAMBackend


def get_backend(config: Optional[Dict[str, Any]] = None) -> WAMBackend:
    backend = WAMBackend(config=config)
    backend.initialize()
    return backend


def embed_image(
    image: Image.Image,
    message: str,
    backend: Optional[WAMBackend] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Image.Image, bool]:
    if backend is None:
        backend = get_backend(config)
    return backend.embed(image, message)


def detect_image(
    image: Image.Image,
    backend: Optional[WAMBackend] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, Optional[str]]:
    if backend is None:
        backend = get_backend(config)
    return backend.detect(image)


def embed_on_path(
    input_path: str,
    output_path: str,
    message: str,
    backend: Optional[WAMBackend] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool]:
    image = Image.open(input_path)
    image_w, ok = embed_image(image, message, backend=backend, config=config)
    if ok:
        image_w.save(output_path)
    return output_path, ok


def detect_on_path(
    input_path: str,
    backend: Optional[WAMBackend] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, Optional[str]]:
    image = Image.open(input_path)
    return detect_image(image, backend=backend, config=config)
