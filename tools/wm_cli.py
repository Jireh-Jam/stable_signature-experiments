from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd

from watermarking.registry import get_model, available_models

# Optional: reuse existing image metrics util
import run_evals as _evals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run watermark detection across models")
    parser.add_argument("--model", type=str, default="stable_signature",
                        help=f"Which model to use. Options: {sorted(available_models().keys())}")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory of images to evaluate")
    parser.add_argument("--img_dir_nw", type=str, default=None,
                        help="Optional directory for non-watermarked images (for SSIM/PSNR/LPIPS)")

    # Common options
    parser.add_argument("--attack_mode", type=str, default="few",
                        choices=["none", "few", "all"], help="Attack suite to apply")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="output/cli")

    # Stable Signature specific
    parser.add_argument("--key_str", type=str, default=None,
                        help="Bitstring key for Stable Signature evaluation (required unless --decode_only)")
    parser.add_argument("--decode_only", action="store_true",
                        help="Only decode bits; do not compare to key")
    parser.add_argument("--msg_decoder_path", type=str, default=None,
                        help="Path to Stable Signature TorchScript/PyTorch decoder")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = get_model(args.model)
    prepare_kwargs = {}
    if args.model == "stable_signature" and args.msg_decoder_path is not None:
        prepare_kwargs["msg_decoder_path"] = args.msg_decoder_path
    model.prepare(**prepare_kwargs)

    rows = model.evaluate_images(
        args.img_dir,
        key_str=args.key_str,
        decode_only=args.decode_only,
        attack_mode=args.attack_mode,
        batch_size=args.batch_size,
    )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "log_stats.csv"), index=False)
    print(f"Saved bit/detection stats to {os.path.join(args.output_dir, 'log_stats.csv')}")

    # Optional image metrics
    if args.img_dir_nw:
        print(">>> Computing image similarity metrics (SSIM/PSNR/Linf)...")
        img_metrics = _evals.get_img_metric(args.img_dir, args.img_dir_nw)
        img_df = pd.DataFrame(img_metrics)
        img_df.to_csv(os.path.join(args.output_dir, 'img_metrics.csv'), index=False)
        print(f"Saved image metrics to {os.path.join(args.output_dir, 'img_metrics.csv')}")


if __name__ == "__main__":
    main()
