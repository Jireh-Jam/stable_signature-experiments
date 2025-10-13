from typing import Any, Dict
import torch.nn as nn

class WatermarkDecoderBuilder:
    """
    Simple interface: given a config dictionary, build and return
    a torch.nn.Module that maps images (B,C,H,W) -> bit logits (B,K).
    """
    def __call__(self, config: Dict[str, Any]) -> nn.Module:  # pragma: no cover
        raise NotImplementedError
