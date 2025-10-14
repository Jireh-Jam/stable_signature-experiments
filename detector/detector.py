"""
Main watermark detection functionality.

This module provides the core watermark detection capabilities,
including single image detection and batch processing.
"""

import logging
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import time

import torch
import numpy as np
from PIL import Image
import pandas as pd

from ..common.config import Config, ModelParams
from ..common.image_utils import load_image, DEFAULT_TRANSFORM, get_image_paths
from .models import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of watermark detection."""
    image_path: str
    detected_message: str
    confidence_scores: np.ndarray
    bit_accuracy: Optional[float] = None
    detection_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


def msg2str(msg: Union[np.ndarray, torch.Tensor]) -> str:
    """Convert message array to binary string."""
    if isinstance(msg, torch.Tensor):
        msg = msg.cpu().numpy()
    return "".join(['1' if el else '0' for el in msg])


def str2msg(s: str) -> List[bool]:
    """Convert binary string to message array."""
    return [True if el == '1' else False for el in s]


class WatermarkDetector:
    """
    Main watermark detection class.
    
    Provides functionality for detecting watermarks in single images
    or batch processing multiple images.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 model_path: Optional[Union[str, Path]] = None,
                 device: Optional[str] = None):
        """
        Initialize the watermark detector.
        
        Args:
            config: Configuration object
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.config = config or Config()
        
        if device is None:
            self.device = self.config.get_device()
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initialized WatermarkDetector on device: {self.device}")
        
        # Initialize model manager
        self.model_manager = ModelManager(self.config.model, self.device)
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        
        self.decoder = None
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained watermark detection model.
        
        Args:
            model_path: Path to model checkpoint
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.decoder = self.model_manager.load_decoder(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def detect_watermark(self, 
                        image: Union[str, Path, Image.Image],
                        expected_message: Optional[str] = None,
                        target_size: Tuple[int, int] = (512, 512)) -> DetectionResult:
        """
        Detect watermark in a single image.
        
        Args:
            image: Input image (path or PIL Image)
            expected_message: Expected watermark message for accuracy calculation
            target_size: Target image size for detection
            
        Returns:
            DetectionResult containing detection information
        """
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            try:
                image = load_image(image, target_size=target_size)
            except Exception as e:
                return DetectionResult(
                    image_path=image_path,
                    detected_message="",
                    confidence_scores=np.array([]),
                    success=False,
                    error_message=f"Failed to load image: {str(e)}"
                )
        else:
            image_path = "memory_image"
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        if self.decoder is None:
            return DetectionResult(
                image_path=image_path,
                detected_message="",
                confidence_scores=np.array([]),
                success=False,
                error_message="No model loaded. Call load_model() first."
            )
        
        try:
            # Preprocess image
            img_tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(self.device)
            
            # Run detection
            with torch.no_grad():
                features = self.decoder(img_tensor)
                confidence_scores = torch.sigmoid(features).cpu().numpy().flatten()
                decoded_bits = features > 0  # Threshold at 0
                decoded_message = msg2str(decoded_bits.squeeze(0).cpu().numpy())
            
            # Calculate bit accuracy if expected message provided
            bit_accuracy = None
            if expected_message is not None:
                if len(expected_message) == len(decoded_message):
                    matches = sum(1 for a, b in zip(expected_message, decoded_message) if a == b)
                    bit_accuracy = matches / len(expected_message)
                else:
                    logger.warning(f"Message length mismatch: expected {len(expected_message)}, got {len(decoded_message)}")
            
            detection_time = time.time() - start_time
            
            return DetectionResult(
                image_path=image_path,
                detected_message=decoded_message,
                confidence_scores=confidence_scores,
                bit_accuracy=bit_accuracy,
                detection_time=detection_time,
                success=True
            )
            
        except Exception as e:
            detection_time = time.time() - start_time
            logger.error(f"Detection failed for {image_path}: {str(e)}")
            
            return DetectionResult(
                image_path=image_path,
                detected_message="",
                confidence_scores=np.array([]),
                detection_time=detection_time,
                success=False,
                error_message=str(e)
            )
    
    def detect_batch(self,
                    image_paths: List[Union[str, Path]],
                    expected_messages: Optional[List[str]] = None,
                    target_size: Tuple[int, int] = (512, 512),
                    batch_size: int = 1) -> List[DetectionResult]:
        """
        Detect watermarks in a batch of images.
        
        Args:
            image_paths: List of image paths
            expected_messages: Optional list of expected messages
            target_size: Target image size for detection
            batch_size: Batch size for processing (currently only supports 1)
            
        Returns:
            List of DetectionResult objects
        """
        if expected_messages and len(expected_messages) != len(image_paths):
            raise ValueError("Number of expected messages must match number of images")
        
        logger.info(f"Processing batch of {len(image_paths)} images")
        
        results = []
        for i, image_path in enumerate(image_paths):
            expected_msg = expected_messages[i] if expected_messages else None
            
            result = self.detect_watermark(
                image_path,
                expected_message=expected_msg,
                target_size=target_size
            )
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_paths)} images")
        
        return results
    
    def process_directory(self,
                         input_dir: Union[str, Path],
                         output_path: Optional[Union[str, Path]] = None,
                         expected_message: Optional[str] = None,
                         recursive: bool = True) -> pd.DataFrame:
        """
        Process all images in a directory and save results to CSV.
        
        Args:
            input_dir: Directory containing images
            output_path: Path to save CSV results (optional)
            expected_message: Expected watermark message for all images
            recursive: Whether to search recursively
            
        Returns:
            DataFrame with detection results
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Get all image paths
        image_paths = get_image_paths(input_dir, recursive=recursive)
        
        if not image_paths:
            logger.warning(f"No images found in {input_dir}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(image_paths)} images in {input_dir}")
        
        # Process images
        results = self.detect_batch(image_paths, expected_messages=[expected_message] * len(image_paths) if expected_message else None)
        
        # Convert to DataFrame
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
            
            # Add confidence scores as separate columns
            if len(result.confidence_scores) > 0:
                for i, score in enumerate(result.confidence_scores):
                    row[f'confidence_bit_{i}'] = score
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved results to {output_path}")
        
        return df
    
    def analyze_detection_quality(self, results: List[DetectionResult]) -> Dict[str, Any]:
        """
        Analyze the quality of detection results.
        
        Args:
            results: List of detection results
            
        Returns:
            Dictionary with analysis metrics
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                'total_images': len(results),
                'successful_detections': 0,
                'success_rate': 0.0,
                'error': 'No successful detections'
            }
        
        # Basic statistics
        analysis = {
            'total_images': len(results),
            'successful_detections': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_detection_time': np.mean([r.detection_time for r in successful_results])
        }
        
        # Bit accuracy statistics (if available)
        accuracy_results = [r for r in successful_results if r.bit_accuracy is not None]
        if accuracy_results:
            accuracies = [r.bit_accuracy for r in accuracy_results]
            analysis.update({
                'avg_bit_accuracy': np.mean(accuracies),
                'std_bit_accuracy': np.std(accuracies),
                'min_bit_accuracy': np.min(accuracies),
                'max_bit_accuracy': np.max(accuracies),
                'perfect_detections': sum(1 for acc in accuracies if acc == 1.0),
                'perfect_detection_rate': sum(1 for acc in accuracies if acc == 1.0) / len(accuracies)
            })
        
        # Confidence score statistics
        all_confidences = []
        for result in successful_results:
            if len(result.confidence_scores) > 0:
                all_confidences.extend(result.confidence_scores)
        
        if all_confidences:
            analysis.update({
                'avg_confidence': np.mean(all_confidences),
                'std_confidence': np.std(all_confidences),
                'min_confidence': np.min(all_confidences),
                'max_confidence': np.max(all_confidences)
            })
        
        # Error analysis
        failed_results = [r for r in results if not r.success]
        if failed_results:
            error_types = {}
            for result in failed_results:
                error_msg = result.error_message or "Unknown error"
                error_types[error_msg] = error_types.get(error_msg, 0) + 1
            analysis['error_types'] = error_types
        
        return analysis
    
    def compare_messages(self, 
                        detected_messages: List[str],
                        expected_message: str) -> Dict[str, Any]:
        """
        Compare detected messages with expected message.
        
        Args:
            detected_messages: List of detected message strings
            expected_message: Expected message string
            
        Returns:
            Comparison statistics
        """
        if not detected_messages:
            return {'error': 'No detected messages provided'}
        
        # Calculate bit-wise accuracy for each message
        accuracies = []
        hamming_distances = []
        
        for detected in detected_messages:
            if len(detected) == len(expected_message):
                matches = sum(1 for a, b in zip(detected, expected_message) if a == b)
                accuracy = matches / len(expected_message)
                hamming_distance = len(expected_message) - matches
            else:
                # Handle length mismatch
                min_len = min(len(detected), len(expected_message))
                matches = sum(1 for a, b in zip(detected[:min_len], expected_message[:min_len]) if a == b)
                accuracy = matches / max(len(detected), len(expected_message))
                hamming_distance = abs(len(detected) - len(expected_message)) + (min_len - matches)
            
            accuracies.append(accuracy)
            hamming_distances.append(hamming_distance)
        
        # Calculate statistics
        comparison_stats = {
            'total_messages': len(detected_messages),
            'expected_message': expected_message,
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'perfect_matches': sum(1 for acc in accuracies if acc == 1.0),
            'perfect_match_rate': sum(1 for acc in accuracies if acc == 1.0) / len(accuracies),
            'avg_hamming_distance': np.mean(hamming_distances),
            'std_hamming_distance': np.std(hamming_distances),
            'min_hamming_distance': np.min(hamming_distances),
            'max_hamming_distance': np.max(hamming_distances)
        }
        
        # Find most common detected message
        message_counts = {}
        for msg in detected_messages:
            message_counts[msg] = message_counts.get(msg, 0) + 1
        
        most_common_message = max(message_counts.items(), key=lambda x: x[1])
        comparison_stats['most_common_message'] = most_common_message[0]
        comparison_stats['most_common_count'] = most_common_message[1]
        comparison_stats['message_diversity'] = len(message_counts)
        
        return comparison_stats
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.decoder is None:
            return {'error': 'No model loaded'}
        
        return {
            'model_type': type(self.decoder).__name__,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.decoder.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.decoder.parameters() if p.requires_grad),
            'model_config': {
                'num_bits': self.config.model.num_bits,
                'decoder_depth': self.config.model.decoder_depth,
                'decoder_channels': self.config.model.decoder_channels
            }
        }