"""
Watermarking pipelines for comprehensive testing and evaluation.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from omegaconf import DictConfig
from PIL import Image, ImageFilter
import io

from .base import BaseWatermarkMethod
from .shared import (
    load_images_from_folder, 
    save_images_to_folder, 
    get_logger,
    setup_logging
)


def run_watermark_pipeline(
    method: BaseWatermarkMethod,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path], 
    message: str,
    config: Optional[DictConfig] = None
) -> bool:
    """
    Run complete watermarking pipeline: embed, transform, detect, evaluate.
    
    Args:
        method: Initialized watermarking method
        input_dir: Directory containing input images
        output_dir: Directory for output results
        message: Watermark message to embed
        config: Configuration object
        
    Returns:
        True if pipeline completed successfully
    """
    logger = get_logger(__name__)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    watermarked_dir = output_dir / "watermarked"
    transformed_dir = output_dir / "transformed"
    results_dir = output_dir / "results"
    
    for dir_path in [watermarked_dir, transformed_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load images
        logger.info("📁 Loading input images...")
        max_images = config.data.max_images if config else -1
        if max_images == -1:
            max_images = None
        
        images = load_images_from_folder(input_dir, max_images=max_images)
        if not images:
            logger.error("No images found in input directory")
            return False
        
        # Step 2: Embed watermarks
        logger.info("🔐 Embedding watermarks...")
        watermarked_images = []
        
        for filename, image in images:
            watermarked_image, success = method.embed_watermark(image, message)
            if success:
                watermarked_images.append((filename, watermarked_image))
            else:
                logger.warning(f"Failed to embed watermark in {filename}")
        
        if not watermarked_images:
            logger.error("Failed to embed watermarks in any images")
            return False
        
        # Save watermarked images
        save_images_to_folder(watermarked_images, watermarked_dir, prefix="wm")
        
        # Step 3: Apply transformations (if configured)
        if config and config.transformations.apply_standard:
            logger.info("🔄 Applying image transformations...")
            apply_transformations(watermarked_images, transformed_dir, config)
        
        # Step 4: Run detection and evaluation
        logger.info("🔍 Running watermark detection...")
        results = run_detection_evaluation(
            method, watermarked_images, transformed_dir, results_dir, config
        )
        
        # Step 5: Generate reports
        logger.info("📊 Generating evaluation reports...")
        generate_evaluation_report(results, results_dir)
        
        logger.info("✅ Watermarking pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        return False


def apply_transformations(
    images: List[tuple], 
    output_dir: Path, 
    config: DictConfig
) -> Dict[str, List[tuple]]:
    """
    Apply various transformations to test watermark robustness.
    
    Args:
        images: List of (filename, image) tuples
        output_dir: Output directory for transformed images
        config: Configuration object
        
    Returns:
        Dictionary mapping transformation names to image lists
    """
    logger = get_logger(__name__)
    
    # Import transformation functions
    try:
        from ..combined_transforms import (
            resize_image, centre_crop, fixed_rotation, 
            gaussian_blur, jpeg_compress, adjust_brightness
        )
    except ImportError:
        # Fallback to basic transformations
        logger.warning("Advanced transformations not available, using basic ones")
        return apply_basic_transformations(images, output_dir, config)
    
    transformations = {}
    
    # Define transformation pipeline based on config
    transform_configs = [
        ("resize_512", lambda inp, out: resize_image(inp, out, size=(512, 512))),
        ("center_crop_224", lambda inp, out: centre_crop(inp, out, size=(224, 224))),
        ("rotation_15deg", lambda inp, out: fixed_rotation(inp, out, degrees=15)),
        ("gaussian_blur", lambda inp, out: gaussian_blur(inp, out, kernel_size=5)),
        ("jpeg_q70", lambda inp, out: jpeg_compress(inp, out, quality=70)),
        ("brightness_boost", lambda inp, out: adjust_brightness(inp, out, brightness_factor=1.3)),
    ]
    
    for transform_name, transform_func in transform_configs:
        logger.info(f"  Applying {transform_name}...")
        transform_dir = output_dir / transform_name
        transform_dir.mkdir(exist_ok=True)
        
        transformed_images = []
        for filename, image in images:
            try:
                # Save original to temp file
                temp_input = transform_dir / f"temp_{filename}"
                from ..shared.io import save_image
                save_image(image, temp_input)
                
                # Apply transformation
                output_path = transform_dir / f"{transform_name}_{filename}"
                transform_func(str(temp_input), str(output_path))
                
                # Load transformed image
                from ..shared.io import load_image
                transformed_image = load_image(output_path)
                if transformed_image:
                    transformed_images.append((filename, transformed_image))
                
                # Clean up temp file
                temp_input.unlink(missing_ok=True)
                
            except Exception as e:
                logger.warning(f"Failed to apply {transform_name} to {filename}: {e}")
        
        transformations[transform_name] = transformed_images
    
    return transformations


def apply_basic_transformations(
    images: List[tuple], 
    output_dir: Path, 
    config: DictConfig
) -> Dict[str, List[tuple]]:
    """
    Apply basic PIL-based transformations as fallback.
    
    Args:
        images: List of (filename, image) tuples
        output_dir: Output directory for transformed images
        config: Configuration object
        
    Returns:
        Dictionary mapping transformation names to image lists
    """
    from PIL import Image, ImageFilter
    import io
    
    transformations = {}
    
    # Basic transformations using PIL
    basic_transforms = [
        ("resize_256", lambda img: img.resize((256, 256), Image.LANCZOS)),
        ("crop_center", lambda img: img.crop((32, 32, img.width-32, img.height-32))),
        ("blur_light", lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))),
        ("jpeg_compress", lambda img: compress_jpeg(img, quality=70)),
    ]
    
    for transform_name, transform_func in basic_transforms:
        transform_dir = output_dir / transform_name
        transform_dir.mkdir(exist_ok=True)
        
        transformed_images = []
        for filename, image in images:
            try:
                transformed_image = transform_func(image)
                transformed_images.append((filename, transformed_image))
            except Exception as e:
                print(f"Error applying {transform_name} to {filename}: {e}")
        
        transformations[transform_name] = transformed_images
        
        # Save transformed images
        save_images_to_folder(
            transformed_images, 
            transform_dir, 
            prefix=transform_name
        )
    
    return transformations


def compress_jpeg(image, quality: int):
    """Apply JPEG compression to image."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def run_detection_evaluation(
    method: BaseWatermarkMethod,
    watermarked_images: List[tuple],
    transformed_dir: Path,
    results_dir: Path,
    config: Optional[DictConfig] = None
) -> Dict[str, Any]:
    """
    Run watermark detection on original and transformed images.
    
    Args:
        method: Watermarking method for detection
        watermarked_images: List of watermarked images
        transformed_dir: Directory containing transformed images
        results_dir: Directory for results
        config: Configuration object
        
    Returns:
        Dictionary containing evaluation results
    """
    logger = get_logger(__name__)
    results = {
        "original": [],
        "transformed": {},
        "summary": {}
    }
    
    # Test detection on original watermarked images
    logger.info("Testing detection on original watermarked images...")
    for filename, image in watermarked_images:
        detected, confidence, message = method.detect_watermark(image)
        results["original"].append({
            "filename": filename,
            "detected": detected,
            "confidence": confidence,
            "message": message
        })
    
    # Test detection on transformed images
    if transformed_dir.exists():
        for transform_dir in transformed_dir.iterdir():
            if transform_dir.is_dir():
                transform_name = transform_dir.name
                logger.info(f"Testing detection on {transform_name} images...")
                
                transform_results = []
                for image_file in transform_dir.glob("*"):
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        from ..shared.io import load_image
                        image = load_image(image_file)
                        if image:
                            detected, confidence, message = method.detect_watermark(image)
                            transform_results.append({
                                "filename": image_file.name,
                                "detected": detected,
                                "confidence": confidence,
                                "message": message
                            })
                
                results["transformed"][transform_name] = transform_results
    
    # Calculate summary statistics
    original_detection_rate = sum(1 for r in results["original"] if r["detected"]) / len(results["original"])
    results["summary"]["original_detection_rate"] = original_detection_rate
    
    for transform_name, transform_results in results["transformed"].items():
        if transform_results:
            detection_rate = sum(1 for r in transform_results if r["detected"]) / len(transform_results)
            results["summary"][f"{transform_name}_detection_rate"] = detection_rate
    
    return results


def generate_evaluation_report(results: Dict[str, Any], output_dir: Path):
    """
    Generate comprehensive evaluation report.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save reports
    """
    logger = get_logger(__name__)
    
    # Save detailed results as JSON
    import json
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary CSV
    import pandas as pd
    
    summary_data = []
    for key, value in results["summary"].items():
        summary_data.append({
            "transformation": key.replace("_detection_rate", ""),
            "detection_rate": value,
            "detection_percentage": f"{value * 100:.1f}%"
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / "summary_results.csv", index=False)
    
    # Generate text report
    with open(output_dir / "evaluation_report.txt", "w") as f:
        f.write("🔐 WATERMARK EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Original Detection Rate: {results['summary']['original_detection_rate']:.1%}\n\n")
        
        f.write("Transformation Results:\n")
        f.write("-" * 30 + "\n")
        for transform_name, transform_results in results["transformed"].items():
            if transform_results:
                detection_rate = sum(1 for r in transform_results if r["detected"]) / len(transform_results)
                f.write(f"{transform_name:20}: {detection_rate:.1%}\n")
    
    logger.info(f"📊 Evaluation report saved to {output_dir}")


# Convenience functions for notebook use
def embed_folder(input_dir: str, output_dir: str, message: str, method_name: str = "stable_signature", max_images: Optional[int] = None):
    """
    Convenience function for embedding watermarks in a folder of images.
    Designed for use in notebooks.
    """
    from . import get_method
    
    method = get_method(method_name)
    if not method.initialize():
        print(f"❌ Failed to initialize {method_name}")
        return []
    
    from .shared import load_images_from_folder
    images = load_images_from_folder(input_dir, max_images=max_images)
    
    results = []
    for filename, image in images:
        watermarked_image, success = method.embed_watermark(image, message)
        results.append({
            "file": filename,
            "success": success,
            "output": str(Path(output_dir) / f"wm_{filename}") if success else None,
            "error": None if success else "Embedding failed"
        })
        
        if success:
            from .shared import save_image
            save_image(watermarked_image, Path(output_dir) / f"wm_{filename}")
    
    return results