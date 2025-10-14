from __future__ import annotations

import argparse
from typing import Optional

from .attack_class import AdvancedWatermarkAttacks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run advanced watermark attack pipelines")
    parser.add_argument('--original', required=True, help='Path to original image')
    parser.add_argument('--watermarked', required=True, help='Path to watermarked image')
    parser.add_argument('--output', default='results', help='Output directory for results')
    parser.add_argument('--attack', default='high_frequency',
                        choices=['high_frequency', 'diffusion_inpainting', 'diffusion_regeneration',
                                 'diffusion_image_to_image', 'adversarial_FGSM', 'adversarial_PGD',
                                 'adversarial_DeepFool', 'all'],
                        help='Attack type to run')
    parser.add_argument('--param', type=float, default=None, help='Attack parameter (threshold, strength, etc.)')
    parser.add_argument('--device', default=None, help='Device to use (cuda or cpu)')
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device if args.device else None
    tool = AdvancedWatermarkAttacks(device=device)

    if args.attack == 'all':
        tool.run_evaluation(args.original, args.watermarked, args.output)
    else:
        tool.run_single_attack(args.original, args.watermarked, args.attack, args.param, args.output)


if __name__ == "__main__":
    main()
