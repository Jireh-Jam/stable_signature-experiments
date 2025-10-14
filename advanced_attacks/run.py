"""
Command-line interface for running watermark attacks.

This script provides a convenient CLI for applying watermark attacks
and running evaluations.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

from ..common.config import load_config, Config
from .attacks import WatermarkAttacker

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('watermark_attacks.log')
        ]
    )


def run_single_attack(args):
    """Run a single attack on an image."""
    attacker = WatermarkAttacker(device=args.device)
    
    logger.info(f"Running {args.attack} attack on {args.watermarked}")
    
    # Parse parameters if provided
    parameters = {}
    if args.parameters:
        for param in args.parameters:
            key, value = param.split('=', 1)
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string if conversion fails
                pass
            parameters[key] = value
    
    result = attacker.apply_attack(
        image=args.watermarked,
        attack_name=args.attack,
        parameters=parameters if parameters else None,
        original_image=args.original if args.original else None
    )
    
    if result.success:
        # Save attacked image
        output_path = Path(args.output) / f"{args.attack}_attacked.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.attacked_image.save(output_path)
        
        logger.info(f"Attack successful! Saved to {output_path}")
        logger.info(f"Execution time: {result.execution_time:.3f}s")
        
        if result.metrics:
            logger.info("Metrics:")
            for metric, value in result.metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
    else:
        logger.error(f"Attack failed: {result.error_message}")
    
    attacker.cleanup()


def run_evaluation(args):
    """Run comprehensive evaluation."""
    config = load_config(args.config) if args.config else None
    attacker = WatermarkAttacker(config=config, device=args.device)
    
    logger.info(f"Running evaluation on {args.watermarked}")
    
    # Parse attack list if provided
    attack_names = None
    if args.attacks:
        attack_names = args.attacks.split(',')
    
    result = attacker.run_comprehensive_evaluation(
        original_image_path=args.original,
        watermarked_image_path=args.watermarked,
        attack_names=attack_names,
        output_dir=args.output
    )
    
    logger.info(f"Evaluation completed in {result.total_time:.2f}s")
    logger.info(f"Success rate: {result.summary_stats.get('success_rate', 0):.2%}")
    
    if 'most_effective_attack' in result.summary_stats:
        logger.info(f"Most effective attack: {result.summary_stats['most_effective_attack']}")
    
    attacker.cleanup()


def run_comparison(args):
    """Run attack comparison across multiple images."""
    config = load_config(args.config) if args.config else None
    attacker = WatermarkAttacker(config=config, device=args.device)
    
    # Parse image pairs
    image_pairs = []
    if args.image_pairs:
        for pair in args.image_pairs:
            original, watermarked = pair.split(',', 1)
            image_pairs.append((original.strip(), watermarked.strip()))
    elif args.image_dir:
        # Auto-discover image pairs in directory
        image_dir = Path(args.image_dir)
        original_dir = image_dir / "original"
        watermarked_dir = image_dir / "watermarked"
        
        if original_dir.exists() and watermarked_dir.exists():
            for orig_file in original_dir.glob("*.png"):
                # Look for corresponding watermarked file
                watermarked_file = watermarked_dir / orig_file.name.replace("_original", "_watermarked")
                if not watermarked_file.exists():
                    watermarked_file = watermarked_dir / orig_file.name
                
                if watermarked_file.exists():
                    image_pairs.append((str(orig_file), str(watermarked_file)))
    
    if not image_pairs:
        logger.error("No image pairs found for comparison")
        return
    
    logger.info(f"Running comparison on {len(image_pairs)} image pairs")
    
    # Parse attack list if provided
    attack_names = None
    if args.attacks:
        attack_names = args.attacks.split(',')
    
    result = attacker.compare_attacks(
        images=image_pairs,
        attack_names=attack_names,
        output_dir=args.output
    )
    
    logger.info("Comparison completed!")
    
    # Print top 5 most effective attacks
    if 'by_effectiveness' in result['rankings']:
        logger.info("Top 5 most effective attacks:")
        for i, attack in enumerate(result['rankings']['by_effectiveness'][:5], 1):
            perf = result['attack_performance'][attack]
            logger.info(f"  {i}. {attack} (success rate: {perf['success_rate']:.2%})")
    
    attacker.cleanup()


def list_attacks(args):
    """List available attacks."""
    attacker = WatermarkAttacker()
    
    attacks = attacker.get_available_attacks(args.category)
    
    if args.category:
        print(f"Available attacks in category '{args.category}':")
    else:
        print("Available attacks:")
    
    for attack in attacks:
        info = attacker.get_attack_info(attack)
        if info:
            print(f"  {attack} - {info['description']} (category: {info['category']})")
        else:
            print(f"  {attack}")
    
    if args.detailed:
        print("\nDetailed information:")
        for attack in attacks:
            info = attacker.get_attack_info(attack)
            if info:
                print(f"\n{attack}:")
                print(f"  Description: {info['description']}")
                print(f"  Category: {info['category']}")
                print(f"  Computational Cost: {info['computational_cost']}")
                print(f"  Effectiveness: {info['effectiveness']}")
                if info['parameters']:
                    print(f"  Parameters: {info['parameters']}")
                if info['requires_models']:
                    print(f"  Required Models: {info['requires_models']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Watermark Attack Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single attack
  python -m advanced_attacks.run single --watermarked image.png --attack high_frequency_filter --output results/

  # Run comprehensive evaluation
  python -m advanced_attacks.run evaluate --original orig.png --watermarked water.png --output results/

  # Compare attacks across multiple images
  python -m advanced_attacks.run compare --image-dir dataset/ --output comparison/

  # List available attacks
  python -m advanced_attacks.run list --detailed
        """
    )
    
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single attack command
    single_parser = subparsers.add_parser('single', help='Run single attack')
    single_parser.add_argument('--watermarked', required=True, help='Path to watermarked image')
    single_parser.add_argument('--original', help='Path to original image (for metrics)')
    single_parser.add_argument('--attack', required=True, help='Attack name to run')
    single_parser.add_argument('--parameters', nargs='*', help='Attack parameters (key=value)')
    single_parser.add_argument('--output', default='output', help='Output directory')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Run comprehensive evaluation')
    eval_parser.add_argument('--original', required=True, help='Path to original image')
    eval_parser.add_argument('--watermarked', required=True, help='Path to watermarked image')
    eval_parser.add_argument('--attacks', help='Comma-separated list of attacks to run')
    eval_parser.add_argument('--output', default='output', help='Output directory')
    
    # Comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare attacks across multiple images')
    compare_parser.add_argument('--image-pairs', nargs='*', help='Image pairs (original,watermarked)')
    compare_parser.add_argument('--image-dir', help='Directory containing original/ and watermarked/ subdirs')
    compare_parser.add_argument('--attacks', help='Comma-separated list of attacks to compare')
    compare_parser.add_argument('--output', default='comparison', help='Output directory')
    
    # List attacks command
    list_parser = subparsers.add_parser('list', help='List available attacks')
    list_parser.add_argument('--category', help='Filter by category')
    list_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'single':
            run_single_attack(args)
        elif args.command == 'evaluate':
            run_evaluation(args)
        elif args.command == 'compare':
            run_comparison(args)
        elif args.command == 'list':
            list_attacks(args)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.log_level == 'DEBUG':
            raise


if __name__ == '__main__':
    main()