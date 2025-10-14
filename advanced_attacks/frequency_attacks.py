"""
Frequency domain attacks for watermark removal.

This module implements attacks that operate in the frequency domain,
targeting high-frequency watermark components.
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FrequencyAttacker:
    """Implements frequency domain attacks on watermarked images."""
    
    def high_frequency_attack(
        self,
        img: np.ndarray,
        threshold_percentile: float = 95,
        filter_strength: float = 0.8
    ) -> np.ndarray:
        """
        Attack watermarks by reducing high frequency components.
        
        This attack assumes watermarks are primarily embedded in high-frequency
        components of the image, which is common for imperceptible watermarks.
        
        Args:
            img: Input watermarked image (BGR format)
            threshold_percentile: Percentile to identify high frequencies (0-100)
            filter_strength: Strength of filtering (0-1, higher = more reduction)
            
        Returns:
            Processed image with reduced high frequency components
        """
        result = img.copy()
        
        # Process each channel separately for color images
        if len(img.shape) > 2:
            for i in range(3):  # BGR channels
                result[:, :, i] = self._filter_channel(
                    img[:, :, i],
                    threshold_percentile,
                    filter_strength
                )
        else:
            # Grayscale image
            result = self._filter_channel(img, threshold_percentile, filter_strength)
        
        return result
    
    def _filter_channel(
        self,
        channel: np.ndarray,
        threshold_percentile: float,
        filter_strength: float
    ) -> np.ndarray:
        """
        Apply frequency filtering to a single channel.
        
        Args:
            channel: Single channel image
            threshold_percentile: Percentile threshold
            filter_strength: Filter strength
            
        Returns:
            Filtered channel
        """
        # Apply FFT
        fft_result = fft2(channel.astype(float))
        fft_shifted = fftshift(fft_result)
        
        # Compute magnitude spectrum
        magnitude = np.abs(fft_shifted)
        
        # Create mask based on threshold
        threshold = np.percentile(magnitude, threshold_percentile)
        high_freq_mask = magnitude > threshold
        
        # Apply filter (reduce high frequencies)
        filtered_fft = fft_shifted.copy()
        filtered_fft[high_freq_mask] *= (1 - filter_strength)
        
        # Inverse FFT
        ifft_result = ifftshift(filtered_fft)
        filtered_channel = np.real(ifft2(ifft_result))
        
        # Normalize and clip values
        filtered_channel = np.clip(filtered_channel, 0, 255).astype(np.uint8)
        
        return filtered_channel
    
    def analyze_frequency_components(
        self,
        img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Analyze the frequency components of an image.
        
        Args:
            img: Input image
            
        Returns:
            Tuple of (magnitude spectrum, high frequency mask, statistics)
        """
        # Convert to grayscale for analysis
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Perform FFT
        fft_result = fft2(gray.astype(float))
        fft_shifted = fftshift(fft_result)
        
        # Compute magnitude spectrum (log scale for visualization)
        magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
        
        # Identify high-frequency components
        threshold = np.percentile(magnitude_spectrum, 95)
        high_frequency_mask = magnitude_spectrum > threshold
        
        # Calculate statistics
        stats = {
            'mean_magnitude': float(np.mean(magnitude_spectrum)),
            'std_magnitude': float(np.std(magnitude_spectrum)),
            'high_freq_ratio': float(np.sum(high_frequency_mask) / high_frequency_mask.size),
            'max_magnitude': float(np.max(magnitude_spectrum)),
            'threshold_95': float(threshold)
        }
        
        return magnitude_spectrum, high_frequency_mask, stats
    
    def adaptive_frequency_attack(
        self,
        img: np.ndarray,
        target_psnr: float = 40.0,
        max_iterations: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        Apply adaptive frequency filtering to achieve target PSNR.
        
        This method iteratively adjusts the filter strength to achieve
        a target PSNR while maximizing watermark removal.
        
        Args:
            img: Input watermarked image
            target_psnr: Target PSNR value
            max_iterations: Maximum iterations for convergence
            
        Returns:
            Tuple of (filtered image, actual filter strength used)
        """
        best_result = img.copy()
        best_strength = 0.0
        
        # Binary search for optimal strength
        low_strength = 0.0
        high_strength = 1.0
        
        for iteration in range(max_iterations):
            strength = (low_strength + high_strength) / 2
            
            # Apply filter
            filtered = self.high_frequency_attack(
                img,
                threshold_percentile=95,
                filter_strength=strength
            )
            
            # Calculate PSNR
            mse = np.mean((img.astype(float) - filtered.astype(float)) ** 2)
            if mse == 0:
                current_psnr = float('inf')
            else:
                current_psnr = 20 * np.log10(255 / np.sqrt(mse))
            
            logger.debug(f"Iteration {iteration}: strength={strength:.3f}, PSNR={current_psnr:.2f}")
            
            # Adjust search range
            if current_psnr > target_psnr:
                # Can be more aggressive
                low_strength = strength
                best_result = filtered
                best_strength = strength
            else:
                # Too aggressive
                high_strength = strength
            
            # Check convergence
            if abs(current_psnr - target_psnr) < 0.5:
                break
        
        return best_result, best_strength
    
    def bandpass_filter_attack(
        self,
        img: np.ndarray,
        low_cutoff: float = 0.1,
        high_cutoff: float = 0.9
    ) -> np.ndarray:
        """
        Apply bandpass filtering to remove specific frequency ranges.
        
        Args:
            img: Input image
            low_cutoff: Low frequency cutoff (0-1)
            high_cutoff: High frequency cutoff (0-1)
            
        Returns:
            Filtered image
        """
        result = img.copy()
        
        # Get image dimensions
        h, w = img.shape[:2]
        center_h, center_w = h // 2, w // 2
        
        # Create frequency mask
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Normalize distances
        max_dist = np.sqrt(center_h**2 + center_w**2)
        dist_normalized = dist_from_center / max_dist
        
        # Create bandpass mask
        mask = np.logical_and(
            dist_normalized >= low_cutoff,
            dist_normalized <= high_cutoff
        ).astype(float)
        
        # Apply to each channel
        if len(img.shape) > 2:
            for i in range(3):
                result[:, :, i] = self._apply_frequency_mask(img[:, :, i], mask)
        else:
            result = self._apply_frequency_mask(img, mask)
        
        return result
    
    def _apply_frequency_mask(
        self,
        channel: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Apply a frequency domain mask to a channel."""
        # FFT
        fft_result = fft2(channel.astype(float))
        fft_shifted = fftshift(fft_result)
        
        # Apply mask
        fft_masked = fft_shifted * mask
        
        # Inverse FFT
        ifft_result = ifftshift(fft_masked)
        filtered = np.real(ifft2(ifft_result))
        
        return np.clip(filtered, 0, 255).astype(np.uint8)