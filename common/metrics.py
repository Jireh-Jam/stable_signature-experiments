"""
Common metrics for evaluating watermark robustness and attack effectiveness.

This module provides metrics for measuring image quality and watermark detection accuracy.
"""

import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def calculate_psnr(
    original: Union[np.ndarray, torch.Tensor],
    modified: Union[np.ndarray, torch.Tensor],
    max_value: float = 255.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        original: Original image
        modified: Modified/attacked image
        max_value: Maximum possible pixel value (255 for uint8)
        
    Returns:
        PSNR value in dB
    """
    # Convert tensors to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(modified, torch.Tensor):
        modified = modified.cpu().numpy()
    
    # Ensure same shape
    if original.shape != modified.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {modified.shape}")
    
    # Calculate MSE
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(max_value / np.sqrt(mse))


def calculate_ssim(
    original: Union[np.ndarray, torch.Tensor],
    modified: Union[np.ndarray, torch.Tensor],
    multichannel: bool = True
) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        original: Original image
        modified: Modified/attacked image
        multichannel: Whether image has multiple channels
        
    Returns:
        SSIM value between -1 and 1 (1 = identical)
    """
    # Convert tensors to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(modified, torch.Tensor):
        modified = modified.cpu().numpy()
    
    # Handle color images
    if len(original.shape) == 3 and multichannel:
        # Convert BGR to grayscale for SSIM
        if original.shape[2] == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            modified_gray = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original[:, :, 0]
            modified_gray = modified[:, :, 0]
    else:
        original_gray = original
        modified_gray = modified
    
    return ssim(original_gray, modified_gray)


def calculate_texture_features(image: np.ndarray) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
    """
    Calculate texture features using LBP and GLCM.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Dictionary containing LBP histogram, statistics, and GLCM features
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # LBP features
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalize
    
    # LBP statistics
    lbp_stats = {
        'skew': float(skew(lbp.flatten())),
        'kurtosis': float(kurtosis(lbp.flatten()))
    }
    
    # Image statistics
    img_stats = {
        'skew': float(skew(gray.flatten())),
        'kurtosis': float(kurtosis(gray.flatten())),
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray))
    }
    
    # GLCM features
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                       symmetric=True, normed=True)
    
    # Properties to extract
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_features = {}
    
    for prop in properties:
        glcm_features[prop] = graycoprops(glcm, prop).flatten()
    
    return {
        'lbp_hist': lbp_hist,
        'lbp_stats': lbp_stats,
        'img_stats': img_stats,
        'glcm_features': glcm_features
    }


def calculate_similarity_metrics(
    features_orig: Dict,
    features_modified: Dict
) -> Dict[str, float]:
    """
    Calculate similarity metrics between original and modified image features.
    
    Args:
        features_orig: Features from original image
        features_modified: Features from modified image
        
    Returns:
        Dictionary of similarity metrics
    """
    # LBP histogram similarity (histogram intersection)
    lbp_similarity = 1 - np.sum(np.abs(
        features_orig['lbp_hist'] - features_modified['lbp_hist']
    )) / 2
    
    # GLCM similarities
    glcm_similarities = {}
    for prop in features_orig['glcm_features'].keys():
        orig = features_orig['glcm_features'][prop]
        modified = features_modified['glcm_features'][prop]
        # Cosine similarity
        similarity = np.dot(orig, modified) / (
            np.linalg.norm(orig) * np.linalg.norm(modified)
        )
        glcm_similarities[prop] = float(similarity)
    
    # Average GLCM similarity
    avg_glcm_similarity = np.mean(list(glcm_similarities.values()))
    
    return {
        'lbp_similarity': float(lbp_similarity),
        'glcm_similarities': glcm_similarities,
        'avg_glcm_similarity': float(avg_glcm_similarity)
    }


def calculate_bit_accuracy(
    original_msg: Union[torch.Tensor, np.ndarray],
    decoded_msg: Union[torch.Tensor, np.ndarray]
) -> Tuple[float, int]:
    """
    Calculate bit accuracy for watermark detection.
    
    Args:
        original_msg: Original watermark message
        decoded_msg: Decoded watermark message
        
    Returns:
        Tuple of (accuracy percentage, number of correct bits)
    """
    # Convert to numpy if needed
    if isinstance(original_msg, torch.Tensor):
        original_msg = original_msg.cpu().numpy()
    if isinstance(decoded_msg, torch.Tensor):
        decoded_msg = decoded_msg.cpu().numpy()
    
    # Ensure boolean arrays
    original_msg = original_msg.astype(bool)
    decoded_msg = decoded_msg.astype(bool)
    
    # Calculate accuracy
    correct_bits = np.sum(original_msg == decoded_msg)
    total_bits = original_msg.size
    accuracy = correct_bits / total_bits
    
    return float(accuracy), int(correct_bits)


def comprehensive_attack_metrics(
    original_image: np.ndarray,
    watermarked_image: np.ndarray,
    attacked_image: np.ndarray,
    original_msg: Optional[Union[torch.Tensor, np.ndarray]] = None,
    decoded_msg: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate comprehensive metrics for watermark attack evaluation.
    
    Args:
        original_image: Original image without watermark
        watermarked_image: Image with watermark
        attacked_image: Image after attack
        original_msg: Original watermark message (optional)
        decoded_msg: Decoded watermark message after attack (optional)
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Image quality metrics
    try:
        # Original vs Watermarked
        metrics['psnr_watermark'] = calculate_psnr(original_image, watermarked_image)
        metrics['ssim_watermark'] = calculate_ssim(original_image, watermarked_image)
        
        # Watermarked vs Attacked
        metrics['psnr_attack'] = calculate_psnr(watermarked_image, attacked_image)
        metrics['ssim_attack'] = calculate_ssim(watermarked_image, attacked_image)
        
    except Exception as e:
        logger.error(f"Error calculating image quality metrics: {str(e)}")
        metrics['psnr_watermark'] = metrics['ssim_watermark'] = -1
        metrics['psnr_attack'] = metrics['ssim_attack'] = -1
    
    # Texture features
    try:
        features_orig = calculate_texture_features(original_image)
        features_watermarked = calculate_texture_features(watermarked_image)
        features_attacked = calculate_texture_features(attacked_image)
        
        # Similarity metrics
        watermark_similarity = calculate_similarity_metrics(
            features_orig, features_watermarked
        )
        attack_similarity = calculate_similarity_metrics(
            features_watermarked, features_attacked
        )
        
        metrics['lbp_similarity_watermark'] = watermark_similarity['lbp_similarity']
        metrics['lbp_similarity_attack'] = attack_similarity['lbp_similarity']
        metrics['glcm_similarity_watermark'] = watermark_similarity['avg_glcm_similarity']
        metrics['glcm_similarity_attack'] = attack_similarity['avg_glcm_similarity']
        
    except Exception as e:
        logger.error(f"Error calculating texture metrics: {str(e)}")
        metrics['lbp_similarity_watermark'] = metrics['lbp_similarity_attack'] = -1
        metrics['glcm_similarity_watermark'] = metrics['glcm_similarity_attack'] = -1
    
    # Watermark detection accuracy
    if original_msg is not None and decoded_msg is not None:
        try:
            accuracy, correct_bits = calculate_bit_accuracy(original_msg, decoded_msg)
            metrics['bit_accuracy'] = accuracy
            metrics['correct_bits'] = correct_bits
        except Exception as e:
            logger.error(f"Error calculating bit accuracy: {str(e)}")
            metrics['bit_accuracy'] = -1
            metrics['correct_bits'] = -1
    
    return metrics