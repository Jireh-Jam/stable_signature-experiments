"""
Command-line interface for watermark detection and evaluation.

This script provides a convenient CLI for detecting watermarks,
batch processing, and evaluation tasks.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

import pandas as pd

from ..common.config import load_config, Config
from .detector import WatermarkDetector

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('watermark_detection.log')
        ]
    )


def detect_single(args):
    """Detect watermark in a single image."""
    detector = WatermarkDetector(device=args.device)
    
    if args.model:
        detector.load_model(args.model)
    else:
        logger.error("Model path is required for detection")
        return
    
    logger.info(f"Detecting watermark in {args.image}")
    
    result = detector.detect_watermark(
        image=args.image,
        expected_message=args.expected_message,
        target_size=(args.size, args.size)
    )
    
    if result.success:
        logger.info(f"Detection successful!")
        logger.info(f"Detected message: {result.detected_message}")
        logger.info(f"Detection time: {result.detection_time:.3f}s")
        
        if result.bit_accuracy is not None:
            logger.info(f"Bit accuracy: {result.bit_accuracy:.2%}")
        
        if len(result.confidence_scores) > 0:
            logger.info(f"Average confidence: {result.confidence_scores.mean():.3f}")
        
        # Save results if output directory provided
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detection result
            result_file = output_dir / "detection_result.txt"
            with open(result_file, 'w') as f:
                f.write(f"Image: {args.image}\n")
                f.write(f"Detected message: {result.detected_message}\n")
                f.write(f"Detection time: {result.detection_time:.3f}s\n")
                if result.bit_accuracy is not None:
                    f.write(f"Bit accuracy: {result.bit_accuracy:.2%}\n")
                if len(result.confidence_scores) > 0:
                    f.write(f"Average confidence: {result.confidence_scores.mean():.3f}\n")
            
            logger.info(f"Results saved to {result_file}")
    else:
        logger.error(f"Detection failed: {result.error_message}")


def detect_batch(args):
    """Detect watermarks in batch."""
    detector = WatermarkDetector(device=args.device)
    
    if args.model:
        detector.load_model(args.model)
    else:
        logger.error("Model path is required for detection")
        return
    
    if args.input_dir:
        # Process directory
        logger.info(f"Processing directory: {args.input_dir}")
        
        df = detector.process_directory(
            input_dir=args.input_dir,
            output_path=args.output,
            expected_message=args.expected_message,
            recursive=args.recursive
        )
        
        if len(df) > 0:
            logger.info(f"Processed {len(df)} images")
            
            # Calculate summary statistics
            successful = df[df['success'] == True]
            if len(successful) > 0:
                logger.info(f"Success rate: {len(successful)/len(df):.2%}")
                
                if 'bit_accuracy' in successful.columns:
                    accuracies = successful['bit_accuracy'].dropna()
                    if len(accuracies) > 0:
                        logger.info(f"Average accuracy: {accuracies.mean():.2%}")
                        logger.info(f"Perfect detections: {(accuracies == 1.0).sum()}/{len(accuracies)}")
        else:
            logger.warning("No images processed")
    
    elif args.image_list:
        # Process from image list file
        image_list_path = Path(args.image_list)
        if not image_list_path.exists():
            logger.error(f"Image list file not found: {args.image_list}")
            return
        
        with open(image_list_path, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(image_paths)} images from list")
        
        # Prepare expected messages
        expected_messages = None
        if args.expected_message:
            expected_messages = [args.expected_message] * len(image_paths)
        
        results = detector.detect_batch(
            image_paths=image_paths,
            expected_messages=expected_messages,
            target_size=(args.size, args.size)
        )
        
        # Save results
        if args.output:
            data = []
            for result in results:
                row = {
                    'image_path': result.image_path,
                    'detected_message': result.detected_message,
                    'bit_accuracy': result.bit_accuracy,
                    'detection_time': result.detection_time,
                    'success': result.success,
                    'error_message': result.error_message
                }
                
                # Add confidence scores
                if len(result.confidence_scores) > 0:
                    for i, score in enumerate(result.confidence_scores):
                        row[f'confidence_bit_{i}'] = score
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
        
        # Print summary
        successful = [r for r in results if r.success]
        logger.info(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results):.2%})")


def evaluate_robustness(args):
    """Evaluate detection robustness against attacks."""
    detector = WatermarkDetector(device=args.device)
    
    if args.model:
        detector.load_model(args.model)
    else:
        logger.error("Model path is required for evaluation")
        return
    
    logger.info("Evaluating detection robustness")
    
    # Get image directories
    original_dir = Path(args.original_dir)
    watermarked_dir = Path(args.watermarked_dir)
    attacked_dir = Path(args.attacked_dir) if args.attacked_dir else None
    
    if not original_dir.exists():
        logger.error(f"Original directory not found: {original_dir}")
        return
    
    if not watermarked_dir.exists():
        logger.error(f"Watermarked directory not found: {watermarked_dir}")
        return
    
    # Find matching image pairs
    watermarked_images = list(watermarked_dir.glob("*.png")) + list(watermarked_dir.glob("*.jpg"))
    
    results = {
        'original': [],
        'watermarked': [],
        'attacked': [] if attacked_dir else None
    }
    
    # Test watermarked images
    logger.info(f"Testing {len(watermarked_images)} watermarked images")
    
    for watermarked_path in watermarked_images:
        result = detector.detect_watermark(
            image=watermarked_path,
            expected_message=args.expected_message
        )
        results['watermarked'].append(result)
    
    # Test attacked images if directory provided
    if attacked_dir and attacked_dir.exists():
        attacked_images = list(attacked_dir.glob("*.png")) + list(attacked_dir.glob("*.jpg"))
        logger.info(f"Testing {len(attacked_images)} attacked images")
        
        for attacked_path in attacked_images:
            result = detector.detect_watermark(
                image=attacked_path,
                expected_message=args.expected_message
            )
            results['attacked'].append(result)
    
    # Analyze results
    watermarked_analysis = detector.analyze_detection_quality(results['watermarked'])
    logger.info("Watermarked Images Analysis:")
    logger.info(f"  Success rate: {watermarked_analysis['success_rate']:.2%}")
    if 'avg_bit_accuracy' in watermarked_analysis:
        logger.info(f"  Average accuracy: {watermarked_analysis['avg_bit_accuracy']:.2%}")
        logger.info(f"  Perfect detections: {watermarked_analysis['perfect_detections']}/{watermarked_analysis['total_images']}")
    
    if results['attacked']:
        attacked_analysis = detector.analyze_detection_quality(results['attacked'])
        logger.info("Attacked Images Analysis:")
        logger.info(f"  Success rate: {attacked_analysis['success_rate']:.2%}")
        if 'avg_bit_accuracy' in attacked_analysis:
            logger.info(f"  Average accuracy: {attacked_analysis['avg_bit_accuracy']:.2%}")
            logger.info(f"  Perfect detections: {attacked_analysis['perfect_detections']}/{attacked_analysis['total_images']}")
        
        # Calculate robustness drop
        if 'avg_bit_accuracy' in watermarked_analysis and 'avg_bit_accuracy' in attacked_analysis:
            robustness_drop = watermarked_analysis['avg_bit_accuracy'] - attacked_analysis['avg_bit_accuracy']
            logger.info(f"  Robustness drop: {robustness_drop:.2%}")
    
    # Save detailed results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save watermarked results
        watermarked_data = []
        for result in results['watermarked']:
            watermarked_data.append({
                'image_path': result.image_path,
                'detected_message': result.detected_message,
                'bit_accuracy': result.bit_accuracy,
                'detection_time': result.detection_time,
                'success': result.success
            })
        
        watermarked_df = pd.DataFrame(watermarked_data)
        watermarked_df.to_csv(output_dir / 'watermarked_results.csv', index=False)
        
        # Save attacked results if available
        if results['attacked']:
            attacked_data = []
            for result in results['attacked']:
                attacked_data.append({
                    'image_path': result.image_path,
                    'detected_message': result.detected_message,
                    'bit_accuracy': result.bit_accuracy,
                    'detection_time': result.detection_time,
                    'success': result.success
                })
            
            attacked_df = pd.DataFrame(attacked_data)
            attacked_df.to_csv(output_dir / 'attacked_results.csv', index=False)
        
        # Save summary report
        with open(output_dir / 'evaluation_summary.txt', 'w') as f:
            f.write("WATERMARK DETECTION ROBUSTNESS EVALUATION\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Watermarked Images:\n")
            for key, value in watermarked_analysis.items():
                f.write(f"  {key}: {value}\n")
            
            if results['attacked']:
                f.write("\nAttacked Images:\n")
                for key, value in attacked_analysis.items():
                    f.write(f"  {key}: {value}\n")
        
        logger.info(f"Evaluation results saved to {output_dir}")


def compare_models(args):
    """Compare multiple models on the same dataset."""
    if not args.models:
        logger.error("At least one model path is required for comparison")
        return
    
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    # Get test images
    test_images = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    logger.info(f"Comparing {len(args.models)} models on {len(test_images)} images")
    
    model_results = {}
    
    for model_path in args.models:
        model_name = Path(model_path).stem
        logger.info(f"Testing model: {model_name}")
        
        # Initialize detector with this model
        detector = WatermarkDetector(device=args.device)
        try:
            detector.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            continue
        
        # Test on all images
        results = []
        for image_path in test_images:
            result = detector.detect_watermark(
                image=image_path,
                expected_message=args.expected_message
            )
            results.append(result)
        
        # Analyze results
        analysis = detector.analyze_detection_quality(results)
        model_results[model_name] = {
            'results': results,
            'analysis': analysis
        }
    
    # Print comparison
    logger.info("\nModel Comparison Results:")
    logger.info("-" * 40)
    
    for model_name, data in model_results.items():
        analysis = data['analysis']
        logger.info(f"{model_name}:")
        logger.info(f"  Success rate: {analysis['success_rate']:.2%}")
        if 'avg_bit_accuracy' in analysis:
            logger.info(f"  Average accuracy: {analysis['avg_bit_accuracy']:.2%}")
            logger.info(f"  Perfect detections: {analysis['perfect_detections']}")
        logger.info(f"  Average time: {analysis['avg_detection_time']:.3f}s")
    
    # Save comparison results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results for each model
        for model_name, data in model_results.items():
            model_data = []
            for result in data['results']:
                model_data.append({
                    'image_path': result.image_path,
                    'detected_message': result.detected_message,
                    'bit_accuracy': result.bit_accuracy,
                    'detection_time': result.detection_time,
                    'success': result.success
                })
            
            model_df = pd.DataFrame(model_data)
            model_df.to_csv(output_dir / f'{model_name}_results.csv', index=False)
        
        # Save comparison summary
        summary_data = []
        for model_name, data in model_results.items():
            analysis = data['analysis']
            summary_data.append({
                'model': model_name,
                'success_rate': analysis['success_rate'],
                'avg_bit_accuracy': analysis.get('avg_bit_accuracy', None),
                'perfect_detections': analysis.get('perfect_detections', None),
                'avg_detection_time': analysis['avg_detection_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        
        logger.info(f"Comparison results saved to {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Watermark Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect watermark in single image
  python -m detector.run detect --image test.png --model ckpts/model.pth

  # Batch process directory
  python -m detector.run batch --input-dir images/ --model ckpts/model.pth --output results.csv

  # Evaluate robustness
  python -m detector.run evaluate --original-dir orig/ --watermarked-dir water/ --attacked-dir attack/ --model ckpts/model.pth

  # Compare models
  python -m detector.run compare --models model1.pth model2.pth --test-dir test/ --output comparison/
        """
    )
    
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single detection command
    detect_parser = subparsers.add_parser('detect', help='Detect watermark in single image')
    detect_parser.add_argument('--image', required=True, help='Path to image')
    detect_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    detect_parser.add_argument('--expected-message', help='Expected watermark message for accuracy calculation')
    detect_parser.add_argument('--size', type=int, default=512, help='Target image size')
    detect_parser.add_argument('--output', help='Output directory for results')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple images')
    batch_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    batch_parser.add_argument('--input-dir', help='Input directory containing images')
    batch_parser.add_argument('--image-list', help='Text file with image paths')
    batch_parser.add_argument('--expected-message', help='Expected watermark message')
    batch_parser.add_argument('--size', type=int, default=512, help='Target image size')
    batch_parser.add_argument('--recursive', action='store_true', help='Search recursively')
    batch_parser.add_argument('--output', help='Output CSV file or directory')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate detection robustness')
    eval_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--original-dir', required=True, help='Directory with original images')
    eval_parser.add_argument('--watermarked-dir', required=True, help='Directory with watermarked images')
    eval_parser.add_argument('--attacked-dir', help='Directory with attacked images')
    eval_parser.add_argument('--expected-message', help='Expected watermark message')
    eval_parser.add_argument('--output', help='Output directory for results')
    
    # Model comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', nargs='+', required=True, help='Paths to model checkpoints')
    compare_parser.add_argument('--test-dir', required=True, help='Directory with test images')
    compare_parser.add_argument('--expected-message', help='Expected watermark message')
    compare_parser.add_argument('--output', help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'detect':
            detect_single(args)
        elif args.command == 'batch':
            detect_batch(args)
        elif args.command == 'evaluate':
            evaluate_robustness(args)
        elif args.command == 'compare':
            compare_models(args)
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