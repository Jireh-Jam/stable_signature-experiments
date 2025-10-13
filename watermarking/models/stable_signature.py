from __future__ import annotations

from typing import Any, Dict, List, Optional

import os
from pathlib import Path

import torch

from watermarking.base import WatermarkModel
from watermarking.registry import register_model

# Reuse helper functions from the repo
import run_evals as _evals


@register_model("stable_signature")
class StableSignatureModel(WatermarkModel):
    def __init__(self) -> None:
        self._decoder = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._default_decoder_path = "models/dec_48b_whit.torchscript.pt"

    @property
    def name(self) -> str:
        return "stable_signature"

    def is_available(self) -> bool:
        return Path(self._default_decoder_path).exists()

    def prepare(self, **kwargs: Any) -> None:
        decoder_path = kwargs.get("msg_decoder_path", self._default_decoder_path)
        if "torchscript" in decoder_path:
            self._decoder = torch.jit.load(decoder_path).to(self._device).eval()
        else:
            # Fallback path: build Python model and load .pth (not recommended for end users)
            msg_decoder = _evals.utils_model.get_hidden_decoder(
                num_bits=kwargs.get("num_bits", 48),
                redundancy=kwargs.get("redundancy", 1),
                num_blocks=kwargs.get("decoder_depth", 8),
                channels=kwargs.get("decoder_channels", 64),
            ).to(self._device)
            ckpt = _evals.utils_model.get_hidden_decoder_ckpt(decoder_path)
            msg_decoder.load_state_dict(ckpt, strict=False)
            msg_decoder.eval()
            self._decoder = msg_decoder

    def _attacks(self, mode: str):
        if mode == "all":
            return {
                'none': lambda x: x,
                'crop_05': lambda x: _evals.utils_img.center_crop(x, 0.5),
                'crop_01': lambda x: _evals.utils_img.center_crop(x, 0.1),
                'rot_25': lambda x: _evals.utils_img.rotate(x, 25),
                'rot_90': lambda x: _evals.utils_img.rotate(x, 90),
                'jpeg_80': lambda x: _evals.utils_img.jpeg_compress(x, 80),
                'jpeg_50': lambda x: _evals.utils_img.jpeg_compress(x, 50),
                'brightness_1p5': lambda x: _evals.utils_img.adjust_brightness(x, 1.5),
                'brightness_2': lambda x: _evals.utils_img.adjust_brightness(x, 2),
                'contrast_1p5': lambda x: _evals.utils_img.adjust_contrast(x, 1.5),
                'contrast_2': lambda x: _evals.utils_img.adjust_contrast(x, 2),
                'saturation_1p5': lambda x: _evals.utils_img.adjust_saturation(x, 1.5),
                'saturation_2': lambda x: _evals.utils_img.adjust_saturation(x, 2),
                'sharpness_1p5': lambda x: _evals.utils_img.adjust_sharpness(x, 1.5),
                'sharpness_2': lambda x: _evals.utils_img.adjust_sharpness(x, 2),
                'resize_05': lambda x: _evals.utils_img.resize(x, 0.5),
                'resize_01': lambda x: _evals.utils_img.resize(x, 0.1),
                'overlay_text': lambda x: _evals.utils_img.overlay_text(x, [76,111,114,101,109,32,73,112,115,117,109]),
                'comb': lambda x: _evals.utils_img.jpeg_compress(
                    _evals.utils_img.adjust_brightness(_evals.utils_img.center_crop(x, 0.5), 1.5), 80
                ),
            }
        elif mode == "few":
            return {
                'none': lambda x: x,
                'crop_01': lambda x: _evals.utils_img.center_crop(x, 0.1),
                'brightness_2': lambda x: _evals.utils_img.adjust_brightness(x, 2),
                'contrast_2': lambda x: _evals.utils_img.adjust_contrast(x, 2),
                'jpeg_50': lambda x: _evals.utils_img.jpeg_compress(x, 50),
                'comb': lambda x: _evals.utils_img.jpeg_compress(
                    _evals.utils_img.adjust_brightness(_evals.utils_img.center_crop(x, 0.5), 1.5), 80
                ),
            }
        else:
            return {'none': lambda x: x}

    def evaluate_images(
        self,
        img_dir: str,
        *,
        key_str: Optional[str] = None,
        decode_only: bool = False,
        attack_mode: str = "few",
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        if self._decoder is None:
            self.prepare()

        attacks = self._attacks(attack_mode)
        if decode_only:
            rows = _evals.get_msgs(img_dir, self._decoder, batch_size=batch_size, attacks=attacks)
        else:
            if key_str is None:
                raise ValueError("key_str is required for bit-accuracy evaluation.")
            key = torch.tensor([c == '1' for c in key_str]).to(self._device)
            rows = _evals.get_bit_accs(img_dir, self._decoder, key, batch_size=batch_size, attacks=attacks)
        return rows
