from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class WatermarkModel(ABC):
    """Abstract base for pluggable watermark models.

    Implementations should encapsulate loading weights and running
    decode/detection on a folder of images, returning row-wise results.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if all dependencies and assets are available locally."""
        pass

    @abstractmethod
    def prepare(self, **kwargs: Any) -> None:
        """Load weights and prepare the detector. Accepts model-specific kwargs."""
        pass

    @abstractmethod
    def evaluate_images(
        self,
        img_dir: str,
        *,
        key_str: Optional[str] = None,
        decode_only: bool = False,
        attack_mode: str = "few",
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Run detection/decoding on images in img_dir and return row dicts.

        Each dict should include at least an 'img' field (row index) and model-specific
        metrics (e.g., bit_acc_* columns, or decoded_* columns when decode_only=True).
        """
        pass
