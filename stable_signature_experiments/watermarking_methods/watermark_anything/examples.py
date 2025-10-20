"""
Lightweight usage examples for the Watermark Anything API.

Usage:
  python -c "from stable_signature_experiments.watermarking_methods.watermark_anything.examples import demo; demo()"
"""
from __future__ import annotations

import os
from typing import Dict, Any
from PIL import Image

from .api import embed_image, detect_image


def demo(tmp_dir: str = "./_wam_demo") -> Dict[str, Any]:
    os.makedirs(tmp_dir, exist_ok=True)
    # Create a tiny sample image
    img = Image.new("RGB", (128, 128), color=(180, 200, 220))

    # Embed a short message
    msg = "101010101010"
    img_w, ok = embed_image(img, msg)

    src_path = os.path.join(tmp_dir, "src.png")
    out_path = os.path.join(tmp_dir, "wm.png")
    img.save(src_path)
    img_w.save(out_path)

    detected, conf, extracted = detect_image(img_w)

    result = {
        "src": os.path.abspath(src_path),
        "wm": os.path.abspath(out_path),
        "embed_success": bool(ok),
        "detected": bool(detected),
        "confidence": float(conf),
        "extracted": extracted,
    }
    print(result)
    return result
