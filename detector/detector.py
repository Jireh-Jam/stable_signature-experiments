"""
Main watermark detector implementation.

This module provides the WatermarkDetector class for detecting watermarks
in images using trained HiDDeN models.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import logging

# Import common utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from common import image_utils, io_utils

from .models import HiddenDecoder
from .utils import msg2str, str2msg, Params

logger = logging.getLogger(__name__)


class WatermarkDetector:
    """
    Watermark detector using HiDDeN models.
    
    This class provides methods to detect and extract watermark messages
    from images using pre-trained decoder models.
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        params: Optional[Params] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the watermark detector.
        
        Args:
            checkpoint_path: Path to model checkpoint
            params: Model parameters (uses defaults if None)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.params = params or Params.default()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing WatermarkDetector on {self.device}")
        
        # Load model
        self.decoder = self._load_decoder()
    
    def _load_decoder(self) -> HiddenDecoder:
        """Load decoder model from checkpoint."""
        # Create decoder model
        decoder = HiddenDecoder(
            num_blocks=self.params.decoder_depth,
            num_bits=self.params.num_bits,
            channels=self.params.decoder_channels
        )
        
        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'encoder_decoder' in state_dict:
            # Full encoder-decoder checkpoint
            full_state_dict = state_dict['encoder_decoder']
            # Remove module prefix if present
            full_state_dict = {k.replace('module.', ''): v for k, v in full_state_dict.items()}
            # Extract decoder weights
            decoder_state_dict = {
                k.replace('decoder.', ''): v 
                for k, v in full_state_dict.items() 
                if 'decoder' in k
            }
        elif 'decoder' in state_dict:
            # Decoder-only checkpoint
            decoder_state_dict = state_dict['decoder']
        else:
            # Assume it's a direct state dict
            decoder_state_dict = state_dict
        
        # Load weights
        decoder.load_state_dict(decoder_state_dict)
        decoder = decoder.to(self.device).eval()
        
        logger.info("Decoder loaded successfully")
        return decoder
    
    def detect(
        self,
        image: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
        return_confidence: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        Detect watermark in an image.
        
        Args:
            image: Input image (path, numpy array, PIL Image, or tensor)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Detected message string, optionally with confidence
        """
        # Load and prepare image
        if isinstance(image, (str, Path)):
            img_tensor, _ = image_utils.prepare_image_for_model(
                str(image), 
                size=(512, 512), 
                device=self.device
            )
        else:
            # Convert to tensor if needed
            if isinstance(image, np.ndarray):
                image = image_utils.cv2_to_pil(image)
            elif isinstance(image, torch.Tensor):
                image = image_utils.tensor_to_image(image)
            
            # Resize if needed
            if image.size != (512, 512):
                image = image.resize((512, 512), Image.Resampling.BICUBIC)
            
            # Convert to tensor
            img_tensor = image_utils.image_to_tensor(image, device=self.device)
            img_tensor = img_tensor.unsqueeze(0)
        
        # Detect watermark
        with torch.no_grad():
            # Forward pass
            logits = self.decoder(img_tensor)  # Shape: (1, num_bits)
            
            # Convert to binary message
            decoded_msg = logits > 0
            
            # Calculate confidence (average absolute logit value)
            confidence = torch.abs(logits).mean().item()
        
        # Convert to string
        msg_str = msg2str(decoded_msg.squeeze(0))
        
        if return_confidence:
            return msg_str, confidence
        return msg_str
    
    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        batch_size: int = 8
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Detect watermarks in a batch of images.
        
        Args:
            images: List of images (paths or arrays)
            batch_size: Batch size for processing
            
        Returns:
            List of detection results with messages and confidences
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = []
            
            # Prepare batch
            for img in batch:
                if isinstance(img, (str, Path)):
                    img_tensor, _ = image_utils.prepare_image_for_model(
                        str(img), size=(512, 512), device=self.device
                    )
                else:
                    # Convert numpy to tensor
                    pil_img = image_utils.cv2_to_pil(img)
                    pil_img = pil_img.resize((512, 512), Image.Resampling.BICUBIC)
                    img_tensor = image_utils.image_to_tensor(pil_img, device=self.device)
                    img_tensor = img_tensor.unsqueeze(0)
                
                batch_tensors.append(img_tensor)
            
            # Stack batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Detect
            with torch.no_grad():
                logits = self.decoder(batch_tensor)
                decoded_msgs = logits > 0
                confidences = torch.abs(logits).mean(dim=1)
            
            # Process results
            for j, (msg, conf) in enumerate(zip(decoded_msgs, confidences)):
                results.append({
                    'message': msg2str(msg),
                    'confidence': conf.item(),
                    'image_index': i + j
                })
        
        return results
    
    def verify_watermark(
        self,
        image: Union[str, Path, np.ndarray],
        expected_message: Union[str, List[bool]],
        threshold: float = 0.75
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Verify if an image contains a specific watermark.
        
        Args:
            image: Input image
            expected_message: Expected watermark message
            threshold: Accuracy threshold for verification (0-1)
            
        Returns:
            Tuple of (is_verified, accuracy, bit_errors_dict)
        """
        # Detect watermark
        detected_msg, confidence = self.detect(image, return_confidence=True)
        
        # Convert expected message to string if needed
        if isinstance(expected_message, list):
            expected_msg = msg2str(expected_message)
        else:
            expected_msg = expected_message
        
        # Calculate bit accuracy
        correct_bits = sum(1 for d, e in zip(detected_msg, expected_msg) if d == e)
        total_bits = len(expected_msg)
        accuracy = correct_bits / total_bits if total_bits > 0 else 0
        
        # Calculate bit error statistics
        bit_errors = {
            'total_bits': total_bits,
            'correct_bits': correct_bits,
            'error_bits': total_bits - correct_bits,
            'accuracy': accuracy,
            'confidence': confidence,
            'bit_error_rate': 1 - accuracy
        }
        
        # Verify
        is_verified = accuracy >= threshold
        
        return is_verified, accuracy, bit_errors
    
    def evaluate_robustness(
        self,
        original_path: Union[str, Path],
        attacked_images: Dict[str, Union[str, Path, np.ndarray]],
        expected_message: Union[str, List[bool]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate watermark robustness against multiple attacks.
        
        Args:
            original_path: Path to original watermarked image
            attacked_images: Dictionary mapping attack names to attacked images
            expected_message: Expected watermark message
            
        Returns:
            Dictionary of evaluation metrics for each attack
        """
        results = {}
        
        # Detect in original
        original_msg, original_conf = self.detect(original_path, return_confidence=True)
        
        # Evaluate each attack
        for attack_name, attacked_img in attacked_images.items():
            try:
                # Verify watermark
                is_verified, accuracy, bit_errors = self.verify_watermark(
                    attacked_img, expected_message
                )
                
                # Store results
                results[attack_name] = {
                    'verified': is_verified,
                    'accuracy': accuracy,
                    'confidence': bit_errors['confidence'],
                    'bit_error_rate': bit_errors['bit_error_rate'],
                    'confidence_drop': original_conf - bit_errors['confidence']
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {attack_name}: {str(e)}")
                results[attack_name] = {
                    'verified': False,
                    'accuracy': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def process_directory(
        self,
        directory: Union[str, Path],
        output_csv: Optional[Union[str, Path]] = None,
        extensions: List[str] = ['.png', '.jpg', '.jpeg']
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
            output_csv: Optional path to save results
            extensions: List of image extensions to process
            
        Returns:
            List of detection results
        """
        directory = Path(directory)
        
        # Find all images
        image_paths = io_utils.get_image_paths(directory, extensions)
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        # Process images
        results = []
        for img_path in image_paths:
            try:
                msg, conf = self.detect(img_path, return_confidence=True)
                results.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'message': msg,
                    'confidence': conf
                })
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                results.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'message': '',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # Save results if requested
        if output_csv:
            io_utils.save_metrics_csv(results, output_csv)
            logger.info(f"Saved results to {output_csv}")
        
        return results