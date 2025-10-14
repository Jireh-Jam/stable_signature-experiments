#!/usr/bin/env python3
"""
Command-line interface for running watermark attacks.

This script provides a CLI for applying various attacks to watermarked images
and evaluating their effectiveness.
"""

import argparse
import logging
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from common import image_utils, metrics, io_utils
from attacks import WatermarkAttacker
from attack_types import AttackConfig, AttackType, get_standard_attack_suite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply watermark attacks and evaluate their effectiveness"
    )
    
    # Input/output arguments
    parser.add_argument(
        "watermarked",
        type=str,
        help="Path to watermarked image"
    )
    parser.add_argument(
        "--original",
        type=str,
        help="Path to original image (for metrics calculation)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (default: ./attack_results)"
    )
    
    # Attack selection
    parser.add_argument(
        "--attack",
        type=str,
        choices=[a.value for a in AttackType],
        help="Specific attack to apply"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all standard attacks"
    )
    
    # Attack parameters
    parser.add_argument(
        "--param",
        type=float,
        help="Primary parameter for the attack (e.g., blur sigma, JPEG quality)"
    )
    parser.add_argument(
        "--param2",
        type=float,
        help="Secondary parameter if needed (e.g., threshold for frequency attack)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A high quality photograph",
        help="Prompt for diffusion-based attacks"
    )
    
    # Other options
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save metrics to CSV file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def create_attack_config(args) -> Optional[AttackConfig]:
    """Create attack configuration from command line arguments."""
    if not args.attack:
        return None
    
    attack_type = AttackType(args.attack)
    
    # Map attack types to configurations
    if attack_type == AttackType.GAUSSIAN_BLUR:
        return AttackConfig.gaussian_blur(
            kernel_size=int(args.param) if args.param else 5,
            sigma=args.param2 if args.param2 else 1.0
        )
    elif attack_type == AttackType.GAUSSIAN_NOISE:
        return AttackConfig.gaussian_noise(std=args.param if args.param else 0.05)
    elif attack_type == AttackType.JPEG_COMPRESSION:
        return AttackConfig.jpeg_compression(quality=int(args.param) if args.param else 80)
    elif attack_type == AttackType.BRIGHTNESS:
        return AttackConfig.brightness(factor=args.param if args.param else 1.2)
    elif attack_type == AttackType.CONTRAST:
        return AttackConfig.contrast(factor=args.param if args.param else 1.2)
    elif attack_type == AttackType.ROTATION:
        return AttackConfig.rotation(degrees=args.param if args.param else 15.0)
    elif attack_type == AttackType.SCALE:
        return AttackConfig.scale(factor=args.param if args.param else 0.5)
    elif attack_type == AttackType.CROP:
        return AttackConfig.crop(ratio=args.param if args.param else 0.5)
    elif attack_type == AttackType.HIGH_FREQUENCY:
        return AttackConfig.high_frequency(
            threshold=args.param if args.param else 95.0,
            strength=args.param2 if args.param2 else 0.8
        )
    elif attack_type == AttackType.DIFFUSION_INPAINTING:
        return AttackConfig.diffusion_inpainting(
            mask_ratio=args.param if args.param else 0.3,
            prompt=args.prompt
        )
    elif attack_type == AttackType.DIFFUSION_REGENERATION:
        return AttackConfig.diffusion_regeneration(
            strength=args.param if args.param else 0.5,
            prompt=args.prompt
        )
    else:
        logger.warning(f"No default configuration for {attack_type}")
        return AttackConfig(attack_type=attack_type, params={})


def main():
    """Main function."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output) if args.output else Path("attack_results")
    output_dir = io_utils.ensure_directory(output_dir)
    
    # Load watermarked image
    logger.info(f"Loading watermarked image: {args.watermarked}")
    watermarked = cv2.imread(args.watermarked)
    if watermarked is None:
        logger.error(f"Failed to load watermarked image: {args.watermarked}")
        return 1
    
    # Load original image if provided
    original = None
    if args.original:
        logger.info(f"Loading original image: {args.original}")
        original = cv2.imread(args.original)
        if original is None:
            logger.warning(f"Failed to load original image: {args.original}")
    
    # Initialize attacker
    attacker = WatermarkAttacker(device=args.device)
    
    # Determine attacks to run
    if args.all:
        logger.info("Running all standard attacks")
        attack_configs = get_standard_attack_suite()
    elif args.attack:
        config = create_attack_config(args)
        if config is None:
            logger.error("Failed to create attack configuration")
            return 1
        attack_configs = {args.attack: config}
    else:
        logger.error("No attack specified. Use --attack or --all")
        return 1
    
    # Run attacks
    all_metrics = []
    
    for attack_name, config in attack_configs.items():
        logger.info(f"\nRunning {attack_name}: {config.description}")
        
        try:
            # Apply attack
            attacked = attacker.attack(watermarked, config)
            
            # Save attacked image
            output_path = output_dir / f"attacked_{attack_name}.png"
            cv2.imwrite(str(output_path), attacked)
            logger.info(f"Saved attacked image to {output_path}")
            
            # Calculate metrics if original is available
            if original is not None:
                attack_metrics = metrics.comprehensive_attack_metrics(
                    original, watermarked, attacked
                )
                attack_metrics['attack_name'] = attack_name
                attack_metrics['attack_type'] = config.attack_type.value
                all_metrics.append(attack_metrics)
                
                # Print key metrics
                logger.info(f"PSNR (watermarked vs attacked): {attack_metrics['psnr_attack']:.2f} dB")
                logger.info(f"SSIM (watermarked vs attacked): {attack_metrics['ssim_attack']:.4f}")
            
        except Exception as e:
            logger.error(f"Error running {attack_name}: {str(e)}")
            continue
    
    # Save metrics if requested
    if args.save_metrics and all_metrics:
        metrics_path = output_dir / "attack_metrics.csv"
        io_utils.save_metrics_csv(all_metrics, metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
    
    logger.info("\nAttack evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())