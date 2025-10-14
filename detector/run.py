#!/usr/bin/env python3
"""
Command-line interface for watermark detection.

This script provides a CLI for detecting watermarks in images using
pre-trained HiDDeN models.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from detector import WatermarkDetector
from utils import Params, str2msg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect watermarks in images using HiDDeN models"
    )
    
    # Input arguments
    parser.add_argument(
        "input",
        type=str,
        help="Input image path or directory"
    )
    
    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpts/hidden_replicate.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON)"
    )
    
    # Detection options
    parser.add_argument(
        "--expected-message",
        type=str,
        help="Expected watermark message for verification"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Accuracy threshold for verification (0-1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing multiple images"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Params:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return Params.from_dict(config_dict)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        return Params.default()


def main():
    """Main function."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        params = load_config(args.config)
    else:
        params = Params.default()
    
    # Initialize detector
    try:
        detector = WatermarkDetector(
            checkpoint_path=args.checkpoint,
            params=params,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {str(e)}")
        return 1
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image detection
        logger.info(f"Detecting watermark in {input_path}")
        
        try:
            if args.expected_message:
                # Verification mode
                is_verified, accuracy, bit_errors = detector.verify_watermark(
                    input_path,
                    args.expected_message,
                    threshold=args.threshold
                )
                
                logger.info(f"Verification result: {'PASSED' if is_verified else 'FAILED'}")
                logger.info(f"Accuracy: {accuracy:.2%}")
                logger.info(f"Bit errors: {bit_errors['error_bits']}/{bit_errors['total_bits']}")
                logger.info(f"Confidence: {bit_errors['confidence']:.4f}")
                
                if args.verbose:
                    detected_msg, _ = detector.detect(input_path, return_confidence=True)
                    logger.info(f"Expected: {args.expected_message}")
                    logger.info(f"Detected: {detected_msg}")
                
            else:
                # Detection mode
                message, confidence = detector.detect(input_path, return_confidence=True)
                
                logger.info(f"Detected message: {message}")
                logger.info(f"Confidence: {confidence:.4f}")
                
                # Save to file if requested
                if args.output:
                    result = [{
                        'image_path': str(input_path),
                        'message': message,
                        'confidence': confidence
                    }]
                    
                    from common import io_utils
                    io_utils.save_metrics_csv(result, args.output)
                    logger.info(f"Saved results to {args.output}")
        
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return 1
    
    elif input_path.is_dir():
        # Directory processing
        logger.info(f"Processing directory: {input_path}")
        
        try:
            results = detector.process_directory(
                input_path,
                output_csv=args.output
            )
            
            logger.info(f"Processed {len(results)} images")
            
            # Print summary
            if results:
                avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
                logger.info(f"Average confidence: {avg_confidence:.4f}")
                
                if args.verbose:
                    # Print first few results
                    for i, result in enumerate(results[:5]):
                        logger.info(f"{result['filename']}: {result['message']} (conf: {result['confidence']:.4f})")
                    if len(results) > 5:
                        logger.info(f"... and {len(results) - 5} more")
        
        except Exception as e:
            logger.error(f"Directory processing failed: {str(e)}")
            return 1
    
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    logger.info("Detection complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())