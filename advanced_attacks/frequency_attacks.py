"""
Frequency domain attacks for watermark removal.

This module implements attacks that operate in the frequency domain,
targeting high-frequency components where watermarks are often embedded.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from ..common.image_utils import pil_to_cv2, cv2_to_pil

logger = logging.getLogger(__name__)


class FrequencyAttacks:
    """Collection of frequency domain watermark attacks."""
    
    @staticmethod
    def high_frequency_filter(image: Image.Image, 
                            threshold_percentile: float = 95,
                            filter_strength: float = 0.8) -> Image.Image:
        """
        Attack watermarks by targeting high frequency components.
        
        This method identifies high-frequency components in the image using FFT
        and reduces their magnitude to potentially remove watermark signals.
        
        Args:
            image: Input watermarked image
            threshold_percentile: Percentile to identify high frequencies (higher = more aggressive)
            filter_strength: Strength of the filter (0-1), higher = more reduction
            
        Returns:
            Processed image with reduced high frequency components
            
        Raises:
            ValueError: If parameters are out of valid range
        """
        if not 0 <= threshold_percentile <= 100:
            raise ValueError("threshold_percentile must be between 0 and 100")
        if not 0 <= filter_strength <= 1:
            raise ValueError("filter_strength must be between 0 and 1")
        
        # Convert to OpenCV format for processing
        img = pil_to_cv2(image)
        result = img.copy()
        
        try:
            # Process each channel for color images
            if len(img.shape) > 2:
                for i in range(3):  # BGR channels
                    channel = img[:, :, i]
                    filtered_channel = FrequencyAttacks._filter_channel(
                        channel, threshold_percentile, filter_strength
                    )
                    result[:, :, i] = filtered_channel
            else:
                # For grayscale images
                result = FrequencyAttacks._filter_channel(
                    img, threshold_percentile, filter_strength
                )
            
            return cv2_to_pil(result)
            
        except Exception as e:
            logger.error(f"Error in high frequency filter: {str(e)}")
            return image
    
    @staticmethod
    def _filter_channel(channel: np.ndarray, 
                       threshold_percentile: float,
                       filter_strength: float) -> np.ndarray:
        """
        Filter a single image channel in frequency domain.
        
        Args:
            channel: Single channel image data
            threshold_percentile: Percentile threshold for high frequencies
            filter_strength: Strength of filtering
            
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
        return np.clip(filtered_channel, 0, 255).astype(np.uint8)
    
    @staticmethod
    def low_pass_filter(image: Image.Image, 
                       cutoff_frequency: float = 0.3) -> Image.Image:
        """
        Apply low-pass filter to remove high frequency components.
        
        Args:
            image: Input image
            cutoff_frequency: Cutoff frequency (0-1, where 1 is Nyquist frequency)
            
        Returns:
            Low-pass filtered image
        """
        if not 0 < cutoff_frequency <= 1:
            raise ValueError("cutoff_frequency must be between 0 and 1")
        
        img = pil_to_cv2(image)
        result = img.copy()
        
        try:
            if len(img.shape) > 2:
                for i in range(3):
                    channel = img[:, :, i]
                    filtered_channel = FrequencyAttacks._apply_low_pass(
                        channel, cutoff_frequency
                    )
                    result[:, :, i] = filtered_channel
            else:
                result = FrequencyAttacks._apply_low_pass(img, cutoff_frequency)
            
            return cv2_to_pil(result)
            
        except Exception as e:
            logger.error(f"Error in low pass filter: {str(e)}")
            return image
    
    @staticmethod
    def _apply_low_pass(channel: np.ndarray, cutoff_frequency: float) -> np.ndarray:
        """Apply low-pass filter to a single channel."""
        h, w = channel.shape
        
        # Create frequency domain coordinates
        u = np.arange(h).reshape(-1, 1) - h // 2
        v = np.arange(w).reshape(1, -1) - w // 2
        
        # Calculate distance from center
        D = np.sqrt(u**2 + v**2)
        
        # Create Butterworth low-pass filter
        D0 = cutoff_frequency * min(h, w) / 2
        n = 2  # Filter order
        H = 1 / (1 + (D / D0)**(2 * n))
        
        # Apply FFT
        fft_result = fft2(channel.astype(float))
        fft_shifted = fftshift(fft_result)
        
        # Apply filter
        filtered_fft = fft_shifted * H
        
        # Inverse FFT
        ifft_result = ifftshift(filtered_fft)
        filtered_channel = np.real(ifft2(ifft_result))
        
        return np.clip(filtered_channel, 0, 255).astype(np.uint8)
    
    @staticmethod
    def notch_filter(image: Image.Image, 
                    center_freq: Tuple[float, float] = (0.1, 0.1),
                    radius: float = 0.05) -> Image.Image:
        """
        Apply notch filter to remove specific frequency components.
        
        Args:
            image: Input image
            center_freq: Center frequency to filter out (normalized coordinates)
            radius: Radius of the notch filter
            
        Returns:
            Notch filtered image
        """
        img = pil_to_cv2(image)
        result = img.copy()
        
        try:
            if len(img.shape) > 2:
                for i in range(3):
                    channel = img[:, :, i]
                    filtered_channel = FrequencyAttacks._apply_notch_filter(
                        channel, center_freq, radius
                    )
                    result[:, :, i] = filtered_channel
            else:
                result = FrequencyAttacks._apply_notch_filter(img, center_freq, radius)
            
            return cv2_to_pil(result)
            
        except Exception as e:
            logger.error(f"Error in notch filter: {str(e)}")
            return image
    
    @staticmethod
    def _apply_notch_filter(channel: np.ndarray, 
                          center_freq: Tuple[float, float],
                          radius: float) -> np.ndarray:
        """Apply notch filter to a single channel."""
        h, w = channel.shape
        
        # Create frequency domain coordinates
        u = np.arange(h).reshape(-1, 1) - h // 2
        v = np.arange(w).reshape(1, -1) - w // 2
        
        # Convert normalized frequencies to actual coordinates
        center_u = center_freq[0] * h
        center_v = center_freq[1] * w
        
        # Calculate distance from notch center
        D = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        
        # Create notch filter (0 at center, 1 elsewhere)
        notch_radius = radius * min(h, w)
        H = np.where(D <= notch_radius, 0, 1)
        
        # Also create symmetric notch (FFT is symmetric)
        D_sym = np.sqrt((u + center_u)**2 + (v + center_v)**2)
        H_sym = np.where(D_sym <= notch_radius, 0, 1)
        H = H * H_sym
        
        # Apply FFT
        fft_result = fft2(channel.astype(float))
        fft_shifted = fftshift(fft_result)
        
        # Apply filter
        filtered_fft = fft_shifted * H
        
        # Inverse FFT
        ifft_result = ifftshift(filtered_fft)
        filtered_channel = np.real(ifft2(ifft_result))
        
        return np.clip(filtered_channel, 0, 255).astype(np.uint8)
    
    @staticmethod
    def analyze_frequency_components(image: Image.Image) -> dict:
        """
        Analyze frequency components of an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing frequency analysis results
        """
        # Convert to grayscale for analysis
        img = pil_to_cv2(image)
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Perform FFT
        fft_result = fft2(gray.astype(float))
        fft_shifted = fftshift(fft_result)
        
        # Compute magnitude spectrum
        magnitude_spectrum = np.abs(fft_shifted)
        log_magnitude = np.log(magnitude_spectrum + 1)
        
        # Calculate statistics
        mean_magnitude = np.mean(magnitude_spectrum)
        std_magnitude = np.std(magnitude_spectrum)
        max_magnitude = np.max(magnitude_spectrum)
        
        # Find dominant frequencies
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Exclude DC component for peak finding
        magnitude_no_dc = magnitude_spectrum.copy()
        magnitude_no_dc[center_h-2:center_h+3, center_w-2:center_w+3] = 0
        
        # Find peaks
        peak_indices = np.unravel_index(
            np.argpartition(magnitude_no_dc.flatten(), -10)[-10:],
            magnitude_no_dc.shape
        )
        
        peaks = []
        for i in range(len(peak_indices[0])):
            u, v = peak_indices[0][i], peak_indices[1][i]
            freq_u = (u - center_h) / h
            freq_v = (v - center_w) / w
            magnitude = magnitude_spectrum[u, v]
            peaks.append({
                'frequency': (freq_u, freq_v),
                'magnitude': float(magnitude),
                'coordinates': (u, v)
            })
        
        # Sort peaks by magnitude
        peaks.sort(key=lambda x: x['magnitude'], reverse=True)
        
        return {
            'magnitude_spectrum': log_magnitude,
            'statistics': {
                'mean_magnitude': float(mean_magnitude),
                'std_magnitude': float(std_magnitude),
                'max_magnitude': float(max_magnitude)
            },
            'dominant_frequencies': peaks[:5],  # Top 5 peaks
            'dc_component': float(magnitude_spectrum[center_h, center_w])
        }