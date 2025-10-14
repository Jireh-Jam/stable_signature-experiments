"""
Diffusion-based attacks for watermark removal.

This module implements attacks using diffusion models like Stable Diffusion
to regenerate or inpaint images, potentially removing embedded watermarks.
"""

import logging
from typing import Optional, Union, List
import tempfile
import os
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image

from ..common.image_utils import pil_to_cv2, cv2_to_pil, ensure_image_size

logger = logging.getLogger(__name__)

# Check for diffusion model availability
try:
    from diffusers import (
        StableDiffusionInpaintPipeline, 
        StableDiffusionImg2ImgPipeline,
        AutoPipelineForImage2Image
    )
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    logger.warning("Diffusion models not available. Install with: pip install diffusers transformers accelerate")

# Check for ReSD pipeline availability
try:
    from .res_pipe import ReSDPipeline
    RESD_AVAILABLE = True
except ImportError:
    RESD_AVAILABLE = False
    logger.warning("ReSDPipeline not available.")


class DiffusionAttacks:
    """Collection of diffusion-based watermark attacks."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize diffusion attack models.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.inpaint_model = None
        self.img2img_model = None
        self.resd_model = None
        
        if DIFFUSION_AVAILABLE:
            self._initialize_models()
        else:
            logger.warning("Diffusion models not available. Attacks will be skipped.")
    
    def _initialize_models(self):
        """Initialize diffusion models lazily."""
        logger.info(f"Initializing diffusion models on {self.device}...")
        
        try:
            # Initialize Stable Diffusion for inpainting
            self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Initialize Stable Diffusion for img2img
            self.img2img_model = AutoPipelineForImage2Image.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            logger.info("Diffusion models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize diffusion models: {str(e)}")
            self.inpaint_model = None
            self.img2img_model = None
    
    def _initialize_resd_model(self):
        """Initialize ReSD model lazily."""
        if not RESD_AVAILABLE:
            return
        
        try:
            self.resd_model = ReSDPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            logger.info("ReSD model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ReSD model: {str(e)}")
            self.resd_model = None
    
    def inpainting_attack(self, 
                         image: Image.Image,
                         prompt: str = "A clean, high-quality photograph",
                         mask_ratio: float = 0.3,
                         guidance_scale: float = 7.5,
                         num_inference_steps: int = 30) -> Image.Image:
        """
        Attack using stable diffusion inpainting.
        
        This method creates a mask over part of the image and uses diffusion
        inpainting to regenerate that region, potentially removing watermarks.
        
        Args:
            image: Input watermarked image
            prompt: Text prompt to guide inpainting
            mask_ratio: Ratio of image area to mask (0-1)
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of denoising steps
            
        Returns:
            Inpainted image
        """
        if not DIFFUSION_AVAILABLE or self.inpaint_model is None:
            logger.warning("Inpainting model not available. Returning original image.")
            return image
        
        try:
            # Ensure image dimensions are compatible (multiple of 8)
            width, height = image.size
            if width % 8 != 0 or height % 8 != 0:
                new_width = (width // 8) * 8
                new_height = (height // 8) * 8
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                width, height = new_width, new_height
            
            # Create mask
            mask = self._create_center_mask(width, height, mask_ratio)
            
            # Run inpainting
            with torch.no_grad():
                result = self.inpaint_model(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Error in inpainting attack: {str(e)}")
            return image
    
    def img2img_attack(self,
                      image: Image.Image,
                      prompt: str = "A clean, high-quality photograph",
                      strength: float = 0.7,
                      guidance_scale: float = 7.5,
                      num_inference_steps: int = 30) -> Image.Image:
        """
        Attack using stable diffusion img2img.
        
        This method uses the input image as a starting point and applies
        diffusion to regenerate it according to the prompt, potentially
        removing watermarks in the process.
        
        Args:
            image: Input watermarked image
            prompt: Text prompt to guide generation
            strength: Strength of transformation (0-1, higher = more change)
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of denoising steps
            
        Returns:
            Regenerated image
        """
        if not DIFFUSION_AVAILABLE or self.img2img_model is None:
            logger.warning("Img2img model not available. Returning original image.")
            return image
        
        try:
            # Ensure image dimensions are compatible
            width, height = image.size
            if width % 8 != 0 or height % 8 != 0:
                new_width = (width // 8) * 8
                new_height = (height // 8) * 8
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Run img2img
            with torch.no_grad():
                result = self.img2img_model(
                    prompt=prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Error in img2img attack: {str(e)}")
            return image
    
    def resd_attack(self,
                   image: Image.Image,
                   prompt: str = "A clean, detailed, high-quality photograph",
                   noise_step: int = 20,
                   strength: float = 0.5,
                   guidance_scale: float = 7.5) -> Image.Image:
        """
        Attack using ReSD (Regeneration Stable Diffusion) approach.
        
        This method uses the ReSD pipeline which adds noise to the image
        and then denoises it with diffusion, potentially removing watermarks.
        
        Args:
            image: Input watermarked image
            prompt: Text prompt to guide regeneration
            noise_step: Number of noise steps to add
            strength: Strength of attack (0-1)
            guidance_scale: Guidance scale for diffusion
            
        Returns:
            Regenerated image
        """
        if not RESD_AVAILABLE:
            logger.warning("ReSD model not available. Falling back to img2img attack.")
            return self.img2img_attack(image, prompt, strength, guidance_scale)
        
        if self.resd_model is None:
            self._initialize_resd_model()
        
        if self.resd_model is None:
            logger.warning("Failed to initialize ReSD model. Returning original image.")
            return image
        
        try:
            # Convert image to tensor format expected by ReSD
            img_array = np.array(image) / 255.0
            img_array = (img_array - 0.5) * 2  # Normalize to [-1, 1]
            
            img_tensor = torch.tensor(
                img_array,
                dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device=self.device
            ).permute(2, 0, 1).unsqueeze(0)
            
            # Generate random seed for reproducibility
            generator = torch.Generator(self.device).manual_seed(42)
            
            # Encode image to latents
            with torch.no_grad():
                latents = self.resd_model.vae.encode(img_tensor).latent_dist
                latents = latents.sample(generator) * self.resd_model.vae.config.scaling_factor
                
                # Add noise
                timestep = torch.tensor([noise_step], dtype=torch.long, device=self.device)
                noise = torch.randn(
                    latents.shape,
                    device=self.device,
                    generator=generator
                )
                
                # Calculate head start step based on strength
                head_start_step = int(50 - max(noise_step // 2, 1) * strength)
                
                # Add noise to latents
                noisy_latents = self.resd_model.scheduler.add_noise(latents, noise, timestep)
                
                # Run the ReSD pipeline
                output = self.resd_model(
                    prompt=prompt,
                    head_start_latents=noisy_latents,
                    head_start_step=head_start_step,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            return output.images[0]
            
        except Exception as e:
            logger.error(f"Error in ReSD attack: {str(e)}")
            return image
    
    def multi_prompt_attack(self,
                           image: Image.Image,
                           prompts: List[str],
                           attack_type: str = "img2img",
                           **kwargs) -> List[Image.Image]:
        """
        Apply diffusion attack with multiple prompts.
        
        Args:
            image: Input image
            prompts: List of prompts to try
            attack_type: Type of attack ('img2img', 'inpainting', 'resd')
            **kwargs: Additional arguments for the attack method
            
        Returns:
            List of attacked images, one for each prompt
        """
        results = []
        
        for prompt in prompts:
            logger.debug(f"Applying {attack_type} attack with prompt: '{prompt}'")
            
            if attack_type == "img2img":
                result = self.img2img_attack(image, prompt, **kwargs)
            elif attack_type == "inpainting":
                result = self.inpainting_attack(image, prompt, **kwargs)
            elif attack_type == "resd":
                result = self.resd_attack(image, prompt, **kwargs)
            else:
                logger.warning(f"Unknown attack type: {attack_type}")
                result = image
            
            results.append(result)
        
        return results
    
    def _create_center_mask(self, width: int, height: int, mask_ratio: float) -> Image.Image:
        """
        Create a centered rectangular mask.
        
        Args:
            width: Image width
            height: Image height
            mask_ratio: Ratio of area to mask
            
        Returns:
            Binary mask image
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate mask dimensions
        mask_w = int(width * mask_ratio)
        mask_h = int(height * mask_ratio)
        
        # Center the mask
        start_w = (width - mask_w) // 2
        start_h = (height - mask_h) // 2
        
        # Create mask
        mask[start_h:start_h + mask_h, start_w:start_w + mask_w] = 255
        
        return Image.fromarray(mask)
    
    def _create_random_mask(self, width: int, height: int, mask_ratio: float) -> Image.Image:
        """
        Create a random mask with specified coverage ratio.
        
        Args:
            width: Image width
            height: Image height
            mask_ratio: Ratio of area to mask
            
        Returns:
            Binary mask image
        """
        total_pixels = width * height
        mask_pixels = int(total_pixels * mask_ratio)
        
        # Create random mask
        mask = np.zeros(total_pixels, dtype=np.uint8)
        mask_indices = np.random.choice(total_pixels, mask_pixels, replace=False)
        mask[mask_indices] = 255
        
        return Image.fromarray(mask.reshape(height, width))
    
    def cleanup(self):
        """Clean up GPU memory by deleting models."""
        if self.inpaint_model is not None:
            del self.inpaint_model
            self.inpaint_model = None
        
        if self.img2img_model is not None:
            del self.img2img_model
            self.img2img_model = None
        
        if self.resd_model is not None:
            del self.resd_model
            self.resd_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleaned up diffusion models")