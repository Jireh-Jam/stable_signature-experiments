from __future__ import annotations

from typing import Dict, Type

from .base import WatermarkModel


_MODEL_REGISTRY: Dict[str, Type[WatermarkModel]] = {}


def register_model(name: str):
    def _decorator(cls: Type[WatermarkModel]):
        _MODEL_REGISTRY[name] = cls
        return cls
    return _decorator


def get_model(name: str) -> WatermarkModel:
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Known: {sorted(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]()


def available_models() -> Dict[str, Type[WatermarkModel]]:
    return dict(_MODEL_REGISTRY)
