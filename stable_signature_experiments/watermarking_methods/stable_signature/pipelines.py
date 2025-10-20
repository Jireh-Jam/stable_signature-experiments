"""
Pipelines for Stable Signature watermarking.

Provides notebook-friendly functions.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from PIL import Image

from .method import StableSignatureMethod


def run_watermark(
    input_path: str,
    output_path: Optional[str] = None,
    message: str = "0" * 48,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool]:
    """Embed a watermark into a single image and save the result.

    Returns the output path and success flag.
    """
    in_path = Path(input_path)
    if output_path is None:
        output_path = str(in_path.with_name(f"wm_{in_path.name}"))

    method = StableSignatureMethod()
    ok_init = method.initialize(config)
    if not ok_init:
        # Graceful fallback: copy image if not initialized
        img = Image.open(in_path)
        img.save(output_path)
        return output_path, False

    img = Image.open(in_path)
    img_w, ok = method.embed_watermark(img, message)
    if ok:
        img_w.save(output_path)
    return output_path, bool(ok)


def detect(
    input_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, Optional[str]]:
    """Detect a watermark in a single image."""
    method = StableSignatureMethod()
    ok_init = method.initialize(config)
    if not ok_init:
        return False, 0.0, None
    img = Image.open(input_path)
    return method.detect_watermark(img)
