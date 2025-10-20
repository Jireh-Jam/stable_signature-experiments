"""
Main CLI entry point for watermarking methods.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .shared import setup_logging, get_logger, load_config, get_default_config
from . import get_method, AVAILABLE_METHODS


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Watermarking Methods CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test watermark robustness
  watermark-test --method stable_signature --input images/ --output results/
  
  # Run with custom config
  watermark-test --config my_config.yaml
  
  # List available methods
  watermark-test --list-methods
        """
    )
    
    parser.add_argument(
        "--method", 
        choices=AVAILABLE_METHODS,
        default="stable_signature",
        help="Watermarking method to use"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input directory containing images"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--message",
        type=str,
        default="test_watermark_48bits",
        help="Watermark message to embed"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images to process"
    )
    
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List available watermarking methods"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def list_methods():
    """List available watermarking methods."""
    print("🔐 Available Watermarking Methods:")
    print("=" * 40)
    
    for method_name in AVAILABLE_METHODS:
        try:
            method = get_method(method_name)
            info = method.get_info()
            print(f"• {method_name}")
            print(f"  Name: {info['name']}")
            print(f"  Class: {info['class']}")
            print()
        except Exception as e:
            print(f"• {method_name} (Error: {e})")
            print()


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(verbose=args.verbose)
    logger = get_logger(__name__)
    
    # Handle special commands
    if args.list_methods:
        list_methods()
        return 0
    
    # Validate required arguments
    if not args.input:
        print("❌ Error: --input directory is required")
        parser.print_help()
        return 1
    
    if not args.output:
        print("❌ Error: --output directory is required") 
        parser.print_help()
        return 1
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        if not config:
            return 1
    else:
        config = get_default_config()
    
    # Update config with command line arguments
    config.watermarking.method = args.method
    config.data.max_images = args.max_images
    
    logger.info(f"Starting watermark testing with method: {args.method}")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        # Initialize watermarking method
        method = get_method(args.method)
        if not method.initialize():
            logger.error(f"Failed to initialize {args.method}")
            return 1
        
        # Run watermarking pipeline
        from .pipelines import run_watermark_pipeline
        
        success = run_watermark_pipeline(
            method=method,
            input_dir=args.input,
            output_dir=args.output,
            message=args.message,
            config=config
        )
        
        if success:
            logger.info("✅ Watermark testing completed successfully")
            return 0
        else:
            logger.error("❌ Watermark testing failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())