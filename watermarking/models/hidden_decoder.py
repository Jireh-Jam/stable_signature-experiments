from typing import Any, Dict
import torch
import torch.nn as nn

import utils_model
from watermarking.registry import register_decoder


@register_decoder("hidden")
def build_hidden_decoder(config: Dict[str, Any]) -> nn.Module:
    """
    Build a HiDDeN-style watermark decoder as a torch.nn.Module.

    Config keys (with defaults):
      - msg_decoder_path: path to checkpoint (TorchScript or .pth). If TorchScript, it is loaded directly.
      - num_bits: 48
      - redundancy: 1
      - decoder_depth: 8
      - decoder_channels: 64
      - device: "auto" | "cpu" | "cuda"
    """
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = config.get("msg_decoder_path", None)

    # If TorchScript checkpoint, prefer direct load
    if ckpt_path and "torchscript" in ckpt_path:
        model = torch.jit.load(ckpt_path).to(device)
        model.eval()
        return model

    num_bits = int(config.get("num_bits", 48))
    redundancy = int(config.get("redundancy", 1))
    depth = int(config.get("decoder_depth", 8))
    channels = int(config.get("decoder_channels", 64))

    model = utils_model.get_hidden_decoder(
        num_bits=num_bits,
        redundancy=redundancy,
        num_blocks=depth,
        channels=channels,
    ).to(device)

    if ckpt_path:
        ckpt = utils_model.get_hidden_decoder_ckpt(ckpt_path)
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model
