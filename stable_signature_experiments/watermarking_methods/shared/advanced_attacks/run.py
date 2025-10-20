from __future__ import annotations

import argparse
import os
from pathlib import Path

from common.logging_utils import get_logger
from .attack_class import AdvancedWatermarkAttacks

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced watermark attacks")
    parser.add_argument("--original", required=True, help="Path to original image")
    parser.add_argument("--watermarked", required=True, help="Path to watermarked image")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument(
        "--attack",
        default="high_frequency",
        choices=[
            "high_frequency",
            "diffusion_inpainting",
            "diffusion_regeneration",
            "diffusion_image_to_image",
            "adversarial_FGSM",
            "adversarial_PGD",
            "adversarial_DeepFool",
            "all",
        ],
        help="Attack to run",
    )
    parser.add_argument("--param", type=float, default=None, help="Attack parameter (threshold/strength/epsilon)")
    parser.add_argument("--device", default=None, help="cuda or cpu; autodetect if not set")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if args.device else ("cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") else ("cuda" if __import__("torch").cuda.is_available() else "cpu"))

    Path(args.output).mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Attack: {args.attack}")

    tool = AdvancedWatermarkAttacks(device=device)
    if args.attack == "all":
        tool.run_evaluation(args.original, args.watermarked, args.output)
    else:
        tool.run_single_attack(args.original, args.watermarked, args.attack, args.param, args.output)


if __name__ == "__main__":
    main()
