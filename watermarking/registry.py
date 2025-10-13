from typing import Callable, Dict

_DecoderRegistry: Dict[str, Callable] = {}


def register_decoder(name: str) -> Callable:
    """Decorator to register a decoder builder under a name."""
    def _decorator(builder: Callable) -> Callable:
        _DecoderRegistry[name] = builder
        return builder
    return _decorator


def get_decoder_builder(name: str) -> Callable:
    if name not in _DecoderRegistry:
        available = ", ".join(sorted(_DecoderRegistry.keys())) or "<none>"
        raise KeyError(f"Unknown decoder '{name}'. Available: {available}")
    return _DecoderRegistry[name]
