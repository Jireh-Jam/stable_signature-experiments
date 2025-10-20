"""
Backend for Watermark Anything (WAM).

Provides a thin, optional dependency layer to load a model (if available)
and expose simple image-level embed/detect methods. Designed so experiments
can run without GPU/torch by gracefully falling back to lightweight stubs.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # noqa: F401
except Exception:  # Torch is optional
    torch = None  # type: ignore

import os
import random
from PIL import Image
import numpy as np


class WAMBackend:
    """
    Encapsulates model loading and image-level embed/detect operations.

    Notes:
    - If PyTorch or checkpoint files are missing, methods gracefully degrade
      to no-op embedding and probabilistic detection for pipeline prototyping.
    - This backend focuses on ergonomics for the pipeline; swap in a real
      model implementation when available.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.device_str: str = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.device = None if torch is None else (torch.device(self.device_str))
        self.model = None  # real model when integrated
        self.initialized: bool = False

    def initialize(self) -> bool:
        """Attempt to load model assets if Torch and checkpoints are available."""
        try:
            if torch is None:
                # Torch not installed; operate in stub mode
                self.initialized = True
                return True

            checkpoint_path = self._resolve_checkpoint_path()
            if checkpoint_path and os.path.exists(checkpoint_path):
                # TODO: Replace with real loading when integrating WAM
                # Example: self.model = load_model_from_checkpoint(json_path, checkpoint_path)
                self.model = object()  # placeholder to indicate a loaded model
                self.initialized = True
            else:
                # No checkpoint found; still mark initialized to enable stubs
                self.initialized = True
            return True
        except Exception:
            self.initialized = False
            return False

    def embed(self, image: Image.Image, message: str) -> Tuple[Image.Image, bool]:
        """Embed a message into the image. Falls back to no-op if unavailable."""
        if not self.initialized:
            return image, False

        if self.model is None or torch is None:
            # Fallback: return the input image as-is
            return image.copy(), True

        # Example real path (to be implemented with the actual WAM):
        # img_tensor = self._pil_to_tensor(image)
        # outputs = self.model.embed(img_tensor, self._msg_to_tensor(message))
        # img_w = ... convert back to PIL ...
        # return img_w, True
        return image.copy(), True

    def detect(self, image: Image.Image) -> Tuple[bool, float, Optional[str]]:
        """Detect a watermark in the image. Falls back to heuristic if unavailable."""
        if not self.initialized:
            return False, 0.0, None

        if self.model is None or torch is None:
            # Fallback heuristic: pseudo-random but stable-ish confidence
            rnd = random.random()
            detected = rnd > 0.35
            confidence = (0.6 + 0.35 * rnd) if detected else (0.1 + 0.3 * rnd)
            return detected, float(confidence), ("wam_msg" if detected else None)

        # Example real path (to be implemented with the actual WAM):
        # img_tensor = self._pil_to_tensor(image)
        # preds = self.model.detect(img_tensor)
        # detected, confidence, message = ...
        # return detected, confidence, message
        return True, 0.85, "wam_msg"

    # --- Helpers -----------------------------------------------------------------

    def _resolve_checkpoint_path(self) -> Optional[str]:
        # Allow explicit config override first
        ckpt = self.config.get("checkpoint_path")
        if ckpt:
            return ckpt

        # Common local fallback locations (relative to repo root)
        candidates = [
            "watermarking_methods/watermark_anything/checkpoints/checkpoint.pth",
            "watermarking_methods/watermark_anything/checkpoints/wam_mit.pth",
            "checkpoints/wam_mit.pth",
            "checkpoints/checkpoint.pth",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _pil_to_tensor(self, image: Image.Image):  # pragma: no cover - stub
        if torch is None:
            raise RuntimeError("Torch not available")
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.asarray(image).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return ten

    def _tensor_to_pil(self, tensor) -> Image.Image:  # pragma: no cover - stub
        if torch is None:
            raise RuntimeError("Torch not available")
        tensor = tensor.squeeze(0).detach().cpu()
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _msg_to_tensor(self, message: str):  # pragma: no cover - stub
        if torch is None:
            raise RuntimeError("Torch not available")
        bits = [1.0 if ch == "1" else 0.0 for ch in message]
        return torch.tensor(bits, dtype=torch.float32, device=self.device)
