"""
Directory-level helpers for embedding and detecting with WAM.
Optimized for simple use in notebooks and scripts.
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image

from .api import get_backend, embed_image, detect_image


def embed_folder(
    input_dir: str,
    output_dir: str,
    message: str,
    max_images: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    backend = get_backend(config)

    image_files = [
        f for f in sorted(os.listdir(input_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if max_images is not None:
        image_files = image_files[:max_images]

    results: List[Dict[str, Any]] = []
    for filename in image_files:
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, f"wm_{filename}")

        try:
            image = Image.open(src)
            image_w, ok = embed_image(image, message, backend=backend)
            if ok:
                image_w.save(dst)
            results.append({
                "file": filename,
                "source": src,
                "output": dst,
                "success": ok,
            })
        except Exception as e:
            results.append({
                "file": filename,
                "source": src,
                "output": dst,
                "success": False,
                "error": str(e),
            })
    return results


def detect_folder(
    dir_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    backend = get_backend(config)
    image_files = [
        f for f in sorted(os.listdir(dir_path))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    results: List[Dict[str, Any]] = []
    for filename in image_files:
        path = os.path.join(dir_path, filename)
        try:
            image = Image.open(path)
            detected, confidence, message = detect_image(image, backend=backend)
            results.append({
                "file": filename,
                "path": path,
                "detected": bool(detected),
                "confidence": float(confidence),
                "message": message,
            })
        except Exception as e:
            results.append({
                "file": filename,
                "path": path,
                "detected": False,
                "confidence": 0.0,
                "message": None,
                "error": str(e),
            })
    return results
