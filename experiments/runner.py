import argparse
import os
from pathlib import Path
import yaml
import torch

from watermarking import get_decoder_builder  # registers models via watermarking.models import

# Reuse robust evaluation utilities from the existing script
import run_evals as evals


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def run_from_config(cfg_path: str) -> None:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg.get("output_dir", "output/")
    _ensure_dir(output_dir)

    # Build decoder if bit evaluation or decoding is requested
    decoder = None
    if cfg.get("eval_bits", False):
        model_cfg = cfg.get("model", {"name": "hidden"})
        name = model_cfg.get("name", "hidden")
        builder = get_decoder_builder(name)
        decoder = builder(model_cfg)
        # Move to current default device
        device = next(decoder.parameters()).device if hasattr(decoder, "parameters") else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        decoder.to(device)
        decoder.eval()

    # 1) Image quality and FID
    if cfg.get("eval_imgs", False):
        img_dir = cfg["img_dir"]
        img_dir_nw = cfg.get("img_dir_nw", None)
        if not img_dir_nw:
            raise ValueError("'img_dir_nw' must be provided when eval_imgs is True")
        save_dir = os.path.join(output_dir, "imgs")
        _ensure_dir(save_dir)

        save_n = int(cfg.get("save_n_imgs", 10))
        if save_n > 0:
            print(f">>> Saving {save_n} example pairs and differences to {save_dir} ...")
            evals.save_imgs(img_dir, img_dir_nw, save_dir, num_imgs=save_n)

        print(">>> Computing image metrics (PSNR, SSIM, Linf) ...")
        num_imgs = cfg.get("num_imgs", None)
        metrics = evals.get_img_metric(img_dir, img_dir_nw, num_imgs=num_imgs)
        import pandas as pd
        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(output_dir, "img_metrics.csv"), index=False)

        if cfg.get("img_dir_fid"):
            print(">>> Computing FID ...")
            fid_w = evals.cached_fid(img_dir, cfg["img_dir_fid"])  # type: ignore[arg-type]
            fid_nw = evals.cached_fid(img_dir_nw, cfg["img_dir_fid"])  # type: ignore[arg-type]
            print(f"FID watermark : {fid_w:.4f}")
            print(f"FID vanilla   : {fid_nw:.4f}")

    # 2) Bit accuracy / decoding
    if cfg.get("eval_bits", False):
        img_dir = cfg["img_dir"]
        batch_size = int(cfg.get("batch_size", 32))
        attack_mode = cfg.get("attack_mode", "few")
        decode_only = bool(cfg.get("decode_only", False))

        # Derive attacks dict using the same logic as the existing script
        if decode_only:
            stats = evals.get_msgs(img_dir, decoder, batch_size=batch_size, attacks=_attack_dict(attack_mode))
        else:
            key_str = cfg.get("key_str")
            if not key_str:
                raise ValueError("'key_str' must be provided unless 'decode_only' is True")
            key = torch.tensor([c == "1" for c in key_str])
            stats = evals.get_bit_accs(img_dir, decoder, key, batch_size=batch_size, attacks=_attack_dict(attack_mode))

        import pandas as pd
        df = pd.DataFrame(stats)
        df.to_csv(os.path.join(output_dir, "log_stats.csv"), index=False)
        print(df.head())


def _attack_dict(mode: str):
    # Mirror the choices used in run_evals.py for familiarity
    import utils_img
    if mode == "all":
        return {
            'none': lambda x: x,
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'contrast_1p5': lambda x: utils_img.adjust_contrast(x, 1.5),
            'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
            'saturation_1p5': lambda x: utils_img.adjust_saturation(x, 1.5),
            'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
            'sharpness_1p5': lambda x: utils_img.adjust_sharpness(x, 1.5),
            'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
            'resize_05': lambda x: utils_img.resize(x, 0.5),
            'resize_01': lambda x: utils_img.resize(x, 0.1),
            'overlay_text': lambda x: utils_img.overlay_text(x, [76,111,114,101,109,32,73,112,115,117,109]),
            'comb': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80),
        }
    elif mode == "few":
        return {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
            'comb': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80),
        }
    else:
        return {'none': lambda x: x}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven watermark evaluation runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run_from_config(args.config)
