"""
CLI for Watermark Anything method.
"""

import argparse
import sys
from pathlib import Path

from ..shared import setup_logging, get_logger, load_image, save_image
from .method import WatermarkAnythingMethod


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for Watermark Anything CLI."""
    parser = argparse.ArgumentParser(
        description="Watermark Anything CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed watermark in single image
  watermark-anything embed input.jpg output.jpg --message "my_watermark"
  
  # Detect watermark in image
  watermark-anything detect watermarked.jpg
  
  # Process folder of images
  watermark-anything embed-folder input_dir/ output_dir/
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed watermark in image")
    embed_parser.add_argument("input", type=Path, help="Input image path")
    embed_parser.add_argument("output", type=Path, help="Output image path")
    embed_parser.add_argument("--message", default="watermark_anything_32b", help="Watermark message")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect watermark in image")
    detect_parser.add_argument("input", type=Path, help="Input image path")
    detect_parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    
    # Embed folder command
    embed_folder_parser = subparsers.add_parser("embed-folder", help="Embed watermarks in folder")
    embed_folder_parser.add_argument("input_dir", type=Path, help="Input directory")
    embed_folder_parser.add_argument("output_dir", type=Path, help="Output directory")
    embed_folder_parser.add_argument("--message", default="watermark_anything_32b", help="Watermark message")
    embed_folder_parser.add_argument("--max-images", type=int, default=100, help="Max images to process")
    
    # Common arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    return parser


def embed_single(args) -> int:
    """Embed watermark in single image."""
    logger = get_logger(__name__)
    
    # Load image
    image = load_image(args.input)
    if not image:
        return 1
    
    # Initialize method
    method = WatermarkAnythingMethod()
    if not method.initialize():
        logger.error("Failed to initialize Watermark Anything method")
        return 1
    
    # Embed watermark
    logger.info(f"Embedding watermark: {args.message}")
    watermarked_image, success = method.embed_watermark(image, args.message)
    
    if not success:
        logger.error("Failed to embed watermark")
        return 1
    
    # Save result
    if save_image(watermarked_image, args.output):
        logger.info(f"✅ Watermarked image saved to {args.output}")
        return 0
    else:
        logger.error("Failed to save watermarked image")
        return 1


def detect_single(args) -> int:
    """Detect watermark in single image."""
    logger = get_logger(__name__)
    
    # Load image
    image = load_image(args.input)
    if not image:
        return 1
    
    # Initialize method
    method = WatermarkAnythingMethod()
    if not method.initialize():
        logger.error("Failed to initialize Watermark Anything method")
        return 1
    
    # Detect watermark
    logger.info(f"Detecting watermark in {args.input}")
    detected, confidence, message = method.detect_watermark(image)
    
    print(f"Detection Result:")
    print(f"  Detected: {'✅ Yes' if detected else '❌ No'}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Message: {message if message else 'None'}")
    print(f"  Threshold: {args.threshold}")
    
    if detected and confidence >= args.threshold:
        logger.info("✅ Watermark detected successfully")
        return 0
    else:
        logger.info("❌ No watermark detected above threshold")
        return 1


def embed_folder(args) -> int:
    """Embed watermarks in folder of images."""
    logger = get_logger(__name__)
    
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize method
    method = WatermarkAnythingMethod()
    if not method.initialize():
        logger.error("Failed to initialize Watermark Anything method")
        return 1
    
    # Process images using the runner module
    from .runner import embed_folder as wam_embed_folder
    
    try:
        results = wam_embed_folder(
            str(args.input_dir),
            str(args.output_dir),
            args.message,
            max_images=args.max_images
        )
        
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"✅ Successfully processed {success_count}/{len(results)} images")
        return 0 if success_count > 0 else 1
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging
    setup_logging(verbose=args.verbose)
    logger = get_logger(__name__)
    
    try:
        if args.command == "embed":
            return embed_single(args)
        elif args.command == "detect":
            return detect_single(args)
        elif args.command == "embed-folder":
            return embed_folder(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())