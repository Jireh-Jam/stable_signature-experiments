"""Transforms registry and composition utilities.

Central place to register, discover, and compose image transformations used
across advanced_attacks and detector. Keeps pipelines consistent and DRY.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Iterable
from PIL import Image
from common.logging_utils import get_logger

logger = get_logger(__name__)


Transform = Callable[[Image.Image], Image.Image]


class TransformsRegistry:
    """Simple name->callable registry for PIL-based transforms.

    All transforms must be pure functions: input Image -> output Image.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Transform] = {}

    def register(self, name: str, fn: Transform, overwrite: bool = False) -> None:
        if not callable(fn):
            raise TypeError(f"Transform '{name}' must be callable")
        if name in self._registry and not overwrite:
            raise ValueError(f"Transform '{name}' already registered")
        self._registry[name] = fn
        logger.debug(f"Registered transform '{name}'")

    def get(self, name: str) -> Transform:
        if name not in self._registry:
            raise KeyError(f"Transform '{name}' not found")
        return self._registry[name]

    def names(self) -> List[str]:
        return sorted(self._registry.keys())

    def build_chain(self, names: Iterable[str]) -> List[Tuple[str, Transform]]:
        chain: List[Tuple[str, Transform]] = []
        for n in names:
            chain.append((n, self.get(n)))
        return chain


# Global default registry
registry = TransformsRegistry()


def apply_chain(image: Image.Image, chain: List[Tuple[str, Transform]]) -> Image.Image:
    """Apply an ordered chain of registered transforms.

    Args:
        image: Input PIL Image
        chain: List of (name, callable) pairs (typically from registry.build_chain)

    Returns:
        Transformed image
    """
    out = image.copy()
    for name, fn in chain:
        out = fn(out)
        logger.debug(f"Applied transform: {name}")
    return out
