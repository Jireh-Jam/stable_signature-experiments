import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from skimage.util import random_noise
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from typing import Optional, Tuple, Dict
from common.logging_utils import setup_logging
import tempfile
import os
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, ifft2, fftshift, ifftshift
# Try to import specific libraries, handle if not available
try:
    from bm3d import bm3d_rgb

    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False
    print("BM3D not available. Install with: pip install bm3d")

try:
    from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor

    COMPRESSAI_AVAILABLE = True
except ImportError:
    COMPRESSAI_AVAILABLE = False
    print("CompressAI not available. Install with: pip install compressai")
# For diffusion models
try:
    from diffusers import StableDiffusionInpaintPipeline,StableDiffusionPipeline, DiffusionPipeline, StableDiffusionImg2ImgPipeline, \
        AutoPipelineForImage2Image
    import torch

    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    print("Diffusion models not available. Install with: pip install diffusers transformers accelerate")
# Import ReSDPipeline separately to avoid circular imports
try:
    from diffusers import StableDiffusionPipeline
    from res_pipe import ReSDPipeline

    RESD_AVAILABLE = True
except ImportError:
    RESD_AVAILABLE = False
    print("ReSDPipeline not available. Make sure it's in your Python path.")


class DirectReSDAttack:
    """
    Implementation of watermark attack using the ReSDPipeline directly.
    This uses the specific capabilities of ReSD for better watermark removal.
    """

    def __init__(self, device: Optional[str] = None,
                 model_path="runwayml/stable-diffusion-v1-5",
                 batch_size: int = 1):
        """
        Initialize the ReSD attack pipeline

        Args:
            device: Device to use (cuda or cpu)
            model_path: Path to pretrained model
            batch_size: Batch size for processing multiple images
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logging(name=__name__)
        self.BATCH_SIZE = batch_size

        self.logger.info(f"Initializing ReSD pipeline on {self.device}...")
        try:
            if not RESD_AVAILABLE:
                print("ReSDPipeline not available. Cannot initialize.")
                self.pipe = None
                return

            # Load the ReSDPipeline model directly
            self.pipe = ReSDPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32
            ).to(device)

            # Set a default noise step
            self.noise_step = 20
            print(f"ReSD pipeline initialized with default noise step {self.noise_step}")
        except Exception as e:
            self.logger.exception(f"Error initializing ReSD pipeline: {str(e)}")
            self.pipe = None

    def attack_image(self, img_path: str, out_path: str, prompt: str = "", noise_step: Optional[int] = None, strength: float = 0.5, return_latents: bool = False):
        """
        Attack a single image using ReSD

        Args:
            img_path: Path to input image
            out_path: Path to save output
            prompt: Text prompt for generation
            noise_step: Noise step (overrides default)
            strength: Strength of attack (0-1)
            return_latents: Whether to return latents

        Returns:
            Attacked image and optionally latents
        """
        if self.pipe is None:
            self.logger.error("ReSD pipeline not initialized successfully. Cannot attack.")
            return None

        # Use provided noise step or default
        noise_step = noise_step if noise_step is not None else self.noise_step

        try:
            # Read image
            img = Image.open(img_path)
            img_array = np.asarray(img) / 255
            img_array = (img_array - 0.5) * 2  # Normalize to [-1, 1]

            # Convert to tensor
            img_tensor = torch.tensor(
                img_array,
                dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device=self.device
            ).permute(2, 0, 1).unsqueeze(0)

            # Generate random seed
            generator = torch.Generator(self.device).manual_seed(42)

            # Encode image to latents
            latents = self.pipe.vae.encode(img_tensor).latent_dist
            latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor

            # Add noise
            timestep = torch.tensor([noise_step], dtype=torch.long, device=self.device)
            noise = torch.randn(
                [1, 4, img_tensor.shape[-2] // 8, img_tensor.shape[-1] // 8],
                device=self.device,
                generator=generator
            )

            # Calculate head start step based on strength
            head_start_step = int(50 - max(noise_step // 2, 1) * strength)

            # Add noise to latents
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

            # Run the ReSD pipeline
            with torch.no_grad():
                output = self.pipe(
                    prompt=prompt,
                    head_start_latents=noisy_latents,
                    head_start_step=head_start_step,
                    guidance_scale=7.5,
                    generator=generator
                )

            # Save the output
            output.images[0].save(out_path)

            # Convert to OpenCV format for further processing if needed
            output_cv = cv2.cvtColor(np.array(output.images[0]), cv2.COLOR_RGB2BGR)

            if return_latents:
                return output_cv, latents.cpu()
            return output_cv

        except Exception as e:
            self.logger.exception(f"Error attacking image {img_path}: {str(e)}")
            return None


class IntegratedWatermarkAttackers:
    """
    Extended class with all watermark attack methods integrated from multiple sources
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize with common parameters and setup"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logging(name=__name__)
        self.logger.info(f"Initialized watermark attackers using device: {self.device}")

        # Initialize diffusion models if available (assuming from original class)
        self.setup_diffusion_models()

    def setup_diffusion_models(self):
        """Setup diffusion models - placeholder to be populated from original class"""
        # This would typically be copied from your AdvancedWatermarkAttacks class
        pass

    # Integrate basic image processing attacks

    def gaussian_blur_attack(self, img, kernel_size: int = 5, sigma: float = 1):
        """Attack using Gaussian blur"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    def gaussian_noise_attack(self, img, std: float = 0.05):
        """Attack using Gaussian noise"""
        # Convert to [0,1] range for noise addition
        img_float = img.astype(np.float32) / 255.0
        # Add noise
        noisy_image = random_noise(img_float, mode='gaussian', var=std ** 2)
        # Clip and convert back to uint8
        noisy_image = np.clip(noisy_image, 0, 1)
        return (noisy_image * 255).astype(np.uint8)

    def jpeg_compression_attack(self, img, quality: int = 80):
        """Attack using JPEG compression"""
        # Need to write to temp file for JPEG compression
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name

        # Convert OpenCV BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(temp_path, "JPEG", quality=quality)

        # Read back and convert to BGR
        compressed_img = cv2.imread(temp_path)

        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass

        return compressed_img

    def brightness_attack(self, img, brightness: float = 0.2):
        """Attack by changing brightness"""
        # Convert OpenCV BGR to RGB PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Apply brightness change
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(brightness)

        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def contrast_attack(self, img, contrast: float = 0.2):
        """Attack by changing contrast"""
        # Convert OpenCV BGR to RGB PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Apply contrast change
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast)

        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def rotation_attack(self, img, degrees: float = 30):
        """Attack by rotating the image"""
        # Convert OpenCV BGR to RGB PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Apply rotation
        img_pil = img_pil.rotate(degrees)

        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def scale_attack(self, img, scale: float = 0.5):
        """Attack by scaling the image down and back up"""
        h, w = img.shape[:2]
        # Scale down
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        # Scale back up to original size
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def crop_attack(self, img, crop_ratio: float = 0.5):
        """Attack by cropping and rescaling"""
        h, w = img.shape[:2]
        # Crop
        crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
        cropped = img[0:crop_h, 0:crop_w]
        # Resize back to original dimensions
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def bm3d_attack(self, img, sigma: float = 0.1):
        """Attack using BM3D denoising"""
        if not BM3D_AVAILABLE:
            print("BM3D not available. Skipping BM3D attack.")
            return img

        # Convert to [0,1] range
        img_float = img.astype(np.float32) / 255.0

        # Apply BM3D denoising
        denoised = bm3d_rgb(img_float, sigma)

        # Convert back to uint8
        return (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
    def diffusion_regeneration_attack(self, img, prompt: str = "A normal image", strength: float = 0.7):
        """Attack using stable diffusion img2img to regenerate the image"""
        if not DIFFUSION_AVAILABLE:
            print("Diffusion models not available. Skipping regeneration attack.")
            return img

        # Store original dimensions
        original_h, original_w = img.shape[:2]

        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Run img2img
        with torch.no_grad():
            output = self.img2img_model(
                prompt=prompt,
                image=img_pil,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]

        # Convert back to OpenCV format
        output_cv = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

        # Ensure output image has the same dimensions as input
        current_h, current_w = output_cv.shape[:2]
        if current_h != original_h or current_w != original_w:
            print(f"Resizing output from {current_w}x{current_h} to {original_w}x{original_h}")
            output_cv = cv2.resize(output_cv, (original_w, original_h), interpolation=cv2.INTER_AREA)

        return output_cv

    def diffusion_image_to_image_attack(self, img, prompt: str = "A normal image", strength: float = 0.7):
        """Attack using stable diffusion img2img to regenerate the image"""
        if not DIFFUSION_AVAILABLE:
            print("Diffusion models not available. Skipping regeneration attack.")
            return img

        # Store original dimensions
        original_h, original_w = img.shape[:2]

        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Run img2img
        with torch.no_grad():
            output = self.img2img_model(
                prompt=prompt,
                image=img_pil,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]

        # Convert back to OpenCV format
        output_cv = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

        # Ensure output image has the same dimensions as input
        current_h, current_w = output_cv.shape[:2]
        if current_h != original_h or current_w != original_w:
            print(f"Resizing output from {current_w}x{current_h} to {original_w}x{original_h}")
            output_cv = cv2.resize(output_cv, (original_w, original_h), interpolation=cv2.INTER_AREA)

        return output_cv

    def diffusion_inpainting_attack(self, img, prompt: str = "A normal image", mask_ratio: float = 0.3, strength: float = 0.75):
        """Attack using stable diffusion inpainting"""
        if not DIFFUSION_AVAILABLE:
            print("Diffusion models not available. Skipping inpainting attack.")
            return img

        # Store original dimensions
        original_h, original_w = img.shape[:2]

        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Create a random mask (or you could target specific areas)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # For simplicity, create a centered rectangular mask
        mask_h, mask_w = int(h * mask_ratio), int(w * mask_ratio)
        start_h, start_w = (h - mask_h) // 2, (w - mask_w) // 2
        mask[start_h:start_h + mask_h, start_w:start_w + mask_w] = 255
        mask_pil = Image.fromarray(mask)

        # Run inpainting
        with torch.no_grad():
            output = self.inpaint_model(
                prompt=prompt,
                image=img_pil,
                mask_image=mask_pil,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]

        # Convert back to OpenCV format
        output_cv = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

        # Ensure output image has the same dimensions as input
        current_h, current_w = output_cv.shape[:2]
        if current_h != original_h or current_w != original_w:
            print(f"Resizing output from {current_w}x{current_h} to {original_w}x{original_h}")
            output_cv = cv2.resize(output_cv, (original_w, original_h), interpolation=cv2.INTER_AREA)

        return output_cv

    def high_frequency_attack(self, img, threshold_percentile: float = 95, filter_strength: float = 0.8):
        """
        Attack watermarks by targeting high frequency components

        Args:
            img: Input watermarked image
            threshold_percentile: Percentile to identify high frequencies (higher = more aggressive)
            filter_strength: Strength of the filter (0-1), higher = more reduction

        Returns:
            Processed image with reduced high frequency components
        """
        # Work with a copy of the image
        result = img.copy()

        # Process each channel for color images
        if len(img.shape) > 2:
            for i in range(3):  # BGR channels
                channel = img[:, :, i]
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
                result[:, :, i] = filtered_channel
        else:
            # For grayscale images
            fft_result = fft2(img.astype(float))
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
            result = np.real(ifft2(ifft_result))

            # Normalize and clip values
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def analyze_frequency_components(self, img, title: str = "Image Frequency Analysis", output_path: Optional[str] = None):
        """
        Analyze and visualize the frequency components of an image

        Args:
            img: Input image
            title: Title for the plot
            output_path: Path to save the visualization

        Returns:
            Tuple of (magnitude_spectrum, high_frequency_mask)
        """
        # Convert to grayscale for FFT analysis
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Perform FFT
        fft_result = fft2(gray.astype(float))
        fft_shifted = fftshift(fft_result)

        # Compute the magnitude spectrum (log scale for better visualization)
        magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)

        # Threshold to identify high-frequency components
        threshold = np.percentile(magnitude_spectrum, 95)
        high_frequency_mask = magnitude_spectrum > threshold

        # Visualize
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) > 2 else gray, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Magnitude Spectrum')
        plt.imshow(magnitude_spectrum, cmap='viridis')
        plt.colorbar(label='Log Magnitude')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('High Frequency Components')
        plt.imshow(high_frequency_mask, cmap='gray')
        plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)

        plt.show()

        return magnitude_spectrum, high_frequency_mask
    def vae_attack(self, img, model_name: str = 'bmshj2018-factorized', quality: int = 1):
        """Attack using VAE compression"""
        if not COMPRESSAI_AVAILABLE:
            print("CompressAI not available. Skipping VAE attack.")
            return img

        # Initialize model
        if model_name == 'bmshj2018-factorized':
            model = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(self.device)
        elif model_name == 'bmshj2018-hyperprior':
            model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(self.device)
        elif model_name == 'mbt2018-mean':
            model = mbt2018_mean(quality=quality, pretrained=True).eval().to(self.device)
        elif model_name == 'mbt2018':
            model = mbt2018(quality=quality, pretrained=True).eval().to(self.device)
        elif model_name == 'cheng2020-anchor':
            model = cheng2020_anchor(quality=quality, pretrained=True).eval().to(self.device)
        else:
            print(f"Unknown model name: {model_name}. Skipping VAE attack.")
            return img

        # Convert to tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(self.device)

        # Apply compression
        with torch.no_grad():
            out = model(img_tensor)
            out['x_hat'].clamp_(0, 1)
            rec = transforms.ToPILImage()(out['x_hat'].squeeze().cpu())

        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(rec), cv2.COLOR_RGB2BGR)

    # Extend run_single_attack to include the new attack types
    def run_single_attack(self, original_path: str, watermarked_path: str, attack_type: str, param: Optional[float | Tuple[float, float]] = None, output_dir: Optional[str] = None, prompt: Optional[str] = None):
        """
        Run a single attack for quick testing with extended attack types

        Args:
            original_path: Path to original image
            watermarked_path: Path to watermarked image
            attack_type: Type of attack to run
            param: Parameter value for attack (can be single value or tuple)
            output_dir: Directory to save output
            prompt: Optional text prompt for diffusion-based attacks
        """
        # Load images
        original_image = cv2.imread(original_path)
        watermarked_image = cv2.imread(watermarked_path)

        if original_image is None or watermarked_image is None:
            print("Error: Could not load one or both images. Please check the file paths.")
            return

        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Apply the selected attack - extending with new attack types
        if attack_type == 'gaussian_blur':
            kernel_size = 5
            sigma = 1
            if param is not None:
                if isinstance(param, tuple) and len(param) == 2:
                    kernel_size, sigma = param
                else:
                    sigma = param

            print(f"Running Gaussian blur attack (kernel={kernel_size}, sigma={sigma})")
            attacked_image = self.gaussian_blur_attack(watermarked_image, kernel_size, sigma)
            param_str = f"kernel={kernel_size},sigma={sigma}"

        elif attack_type == 'gaussian_noise':
            std = param if param is not None else 0.05
            print(f"Running Gaussian noise attack (std={std})")
            attacked_image = self.gaussian_noise_attack(watermarked_image, std)
            param_str = f"std={std}"

        elif attack_type == 'jpeg':
            quality = param if param is not None else 80
            print(f"Running JPEG compression attack (quality={quality})")
            attacked_image = self.jpeg_compression_attack(watermarked_image, quality)
            param_str = f"quality={quality}"

        elif attack_type == 'brightness':
            brightness = param if param is not None else 0.2
            print(f"Running brightness attack (brightness={brightness})")
            attacked_image = self.brightness_attack(watermarked_image, brightness)
            param_str = f"brightness={brightness}"

        elif attack_type == 'contrast':
            contrast = param if param is not None else 0.2
            print(f"Running contrast attack (contrast={contrast})")
            attacked_image = self.contrast_attack(watermarked_image, contrast)
            param_str = f"contrast={contrast}"

        elif attack_type == 'rotation':
            degrees = param if param is not None else 30
            print(f"Running rotation attack (degrees={degrees})")
            attacked_image = self.rotation_attack(watermarked_image, degrees)
            param_str = f"degrees={degrees}"

        elif attack_type == 'scale':
            scale = param if param is not None else 0.5
            print(f"Running scale attack (scale={scale})")
            attacked_image = self.scale_attack(watermarked_image, scale)
            param_str = f"scale={scale}"

        elif attack_type == 'crop':
            crop_ratio = param if param is not None else 0.5
            print(f"Running crop attack (crop_ratio={crop_ratio})")
            attacked_image = self.crop_attack(watermarked_image, crop_ratio)
            param_str = f"crop={crop_ratio}"

        elif attack_type == 'bm3d':
            sigma = param if param is not None else 0.1
            print(f"Running BM3D denoising attack (sigma={sigma})")
            attacked_image = self.bm3d_attack(watermarked_image, sigma)
            param_str = f"sigma={sigma}"

        elif attack_type == 'vae':
            model_name = 'bmshj2018-factorized'
            quality = 1

            if param is not None:
                if isinstance(param, tuple) and len(param) == 2:
                    model_name, quality = param
                else:
                    quality = param

            print(f"Running VAE compression attack (model={model_name}, quality={quality})")
            attacked_image = self.vae_attack(watermarked_image, model_name, quality)
            param_str = f"model={model_name},quality={quality}"

        # Keep existing attack types (assuming they exist in the original class)
        elif attack_type == 'diffusion_inpainting':
            mask_ratio = param if param is not None else 0.3
            print(f"Running diffusion inpainting attack (mask ratio={mask_ratio})")
            attacked_image = self.diffusion_inpainting_attack(
                watermarked_image,
                prompt=prompt or "A clean, high-quality photograph",
                mask_ratio=mask_ratio
            )
            param_str = f"mask={mask_ratio}"

        elif attack_type == 'diffusion_regeneration':
            strength = param if param is not None else 0.5
            print(f"Running diffusion regeneration attack (strength={strength})")
            attacked_image = self.diffusion_regeneration_attack(
                watermarked_image,
                prompt=prompt or "A clean, high-quality photograph",
                strength=strength
            )
            param_str = f"strength={strength}"

        elif attack_type == 'diffusion_resd_direct':
            # Parse parameters
            strength = 0.5
            noise_step = 20

            if param is not None:
                if isinstance(param, tuple) and len(param) == 2:
                    strength, noise_step = param
                else:
                    strength = param

            print(f"Running direct ReSD attack (strength={strength}, noise_step={noise_step})")

            # Use default prompt if none provided
            default_prompt = "A clean, detailed, high-quality photograph"

            # Create a temporary ReSD attacker
            resd_attacker = DirectReSDAttack(device=self.device)

            # Use temporary files to perform the attack
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_in, \
                    tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_out:
                input_path = tmp_in.name
                output_path = tmp_out.name

            # Save and process
            cv2.imwrite(input_path, watermarked_image)
            attacked_image = resd_attacker.attack_image(
                input_path,
                output_path,
                prompt=prompt or default_prompt,
                noise_step=noise_step,
                strength=strength
            )

            # If attack failed, read from file or return original
            if attacked_image is None:
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    attacked_image = cv2.imread(output_path)
                else:
                    attacked_image = watermarked_image.copy()
                    print("Attack failed. Using original image.")

            # Clean up
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass

            param_str = f"strength={strength},noise={noise_step}"

        elif attack_type.startswith('adversarial_'):
            adv_type = attack_type.split('_')[1] if '_' in attack_type else 'FGSM'
            epsilon = param if param is not None else 0.03
            print(f"Running adversarial attack ({adv_type}, epsilon={epsilon})")
            attacked_image = self.adversarial_attack(
                watermarked_image, attack_type=adv_type, epsilon=epsilon
            )
            param_str = f"epsilon={epsilon}"

        elif attack_type == 'high_frequency':
            # For high frequency attack, param can be a tuple (threshold, strength) or just threshold
            if param is None:
                # Default parameters
                threshold = 95
                strength = 0.8
            elif isinstance(param, tuple) and len(param) == 2:
                threshold, strength = param
            else:
                threshold = param
                strength = 0.8

            print(f"Running high frequency attack (threshold={threshold}, strength={strength})")

            # Analyze frequency components first if output directory specified
            if output_dir:
                self.analyze_frequency_components(
                    watermarked_image,
                    "Frequency Analysis of Watermarked Image",
                    os.path.join(output_dir, "frequency_analysis.png")
                )

            # Apply the attack
            attacked_image = self.high_frequency_attack(
                watermarked_image,
                threshold_percentile=threshold,
                filter_strength=strength
            )
            param_str = f"thresh={threshold},str={strength}"

        else:
            print(f"Unknown attack type: {attack_type}")
            return

        # Evaluate and visualize results - assuming plot_attack_comparison exists from original class
        try:
            metrics = self.plot_attack_comparison(
                original_image, watermarked_image, attacked_image,
                attack_type, param_str, output_dir
            )
        except Exception as e:
            print(f"Error in visualizing results: {str(e)}")
            metrics = None

        # Save attacked image if output directory is provided
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"attacked_{attack_type}_{param_str}.png"), attacked_image)

        return metrics, attacked_image

    # Function to add ReSD attack capability to an existing instance

    def run_comprehensive_evaluation(self, original_path: str, watermarked_path: str, output_dir: Optional[str] = None):
        """
        Run evaluation with all available attack methods

        Args:
            original_path: Path to original image
            watermarked_path: Path to watermarked image
            output_dir: Directory to save results
        """
        # Load images
        original_image = cv2.imread(original_path)
        watermarked_image = cv2.imread(watermarked_path)

        if original_image is None or watermarked_image is None:
            print("Error: Could not load one or both images. Please check the file paths.")
            return

        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = {}

        # Basic image processing attacks
        print("\n=== Running Basic Image Processing Attacks ===")

        # Gaussian Blur attacks
        kernels = [3, 5, 7]
        sigmas = [0.5, 1, 2]
        for kernel in kernels:
            for sigma in sigmas:
                print(f"\nRunning Gaussian blur attack (kernel={kernel}, sigma={sigma})")
                metrics, _ = self.run_single_attack(
                    original_path, watermarked_path, 'gaussian_blur', (kernel, sigma), output_dir
                )
                results[f"gaussian_blur_k{kernel}_s{sigma}"] = metrics

        # Gaussian Noise attacks
        stds = [0.01, 0.05, 0.1]
        for std in stds:
            print(f"\nRunning Gaussian noise attack (std={std})")
            metrics, _ = self.run_single_attack(
                original_path, watermarked_path, 'gaussian_noise', std, output_dir
            )
            results[f"gaussian_noise_std{std}"] = metrics

        # JPEG Compression attacks
        qualities = [10, 30, 50, 70, 90]
        for quality in qualities:
            print(f"\nRunning JPEG compression attack (quality={quality})")
            metrics, _ = self.run_single_attack(
                original_path, watermarked_path, 'jpeg', quality, output_dir
            )
            results[f"jpeg_quality{quality}"] = metrics

        # Brightness attacks
        brightness_values = [0.2, 0.5, 0.8, 1.2, 1.5]
        for brightness in brightness_values:
            print(f"\nRunning brightness attack (brightness={brightness})")
            metrics, _ = self.run_single_attack(
                original_path, watermarked_path, 'brightness', brightness, output_dir
            )
            results[f"brightness_{brightness}"] = metrics

        # Contrast attacks
        contrast_values = [0.2, 0.5, 0.8, 1.2, 1.5]
        for contrast in contrast_values:
            print(f"\nRunning contrast attack (contrast={contrast})")
            metrics, _ = self.run_single_attack(
                original_path, watermarked_path, 'contrast', contrast, output_dir
            )
            results[f"contrast_{contrast}"] = metrics

        # Rotation attacks
        degrees = [5, 15, 30, 45, 90]
        for degree in degrees:
            print(f"\nRunning rotation attack (degrees={degree})")
            metrics, _ = self.run_single_attack(
                original_path, watermarked_path, 'rotation', degree, output_dir
            )
            results[f"rotation_{degree}"] = metrics

        # Scale attacks
        scales = [0.1, 0.3, 0.5, 0.8]
        for scale in scales:
            print(f"\nRunning scale attack (scale={scale})")
            metrics, _ = self.run_single_attack(
                original_path, watermarked_path, 'scale', scale, output_dir
            )
            results[f"scale_{scale}"] = metrics

        # Crop attacks
        crop_ratios = [0.3, 0.5, 0.7, 0.9]
        for crop_ratio in crop_ratios:
            print(f"\nRunning crop attack (crop_ratio={crop_ratio})")
            metrics, _ = self.run_single_attack(
                original_path, watermarked_path, 'crop', crop_ratio, output_dir
            )
            results[f"crop_{crop_ratio}"] = metrics

        # Advanced attacks if available

        # BM3D denoising if available
        if BM3D_AVAILABLE:
            print("\n=== Running BM3D Denoising Attacks ===")
            sigmas = [0.05, 0.1, 0.2]
            for sigma in sigmas:
                print(f"\nRunning BM3D attack (sigma={sigma})")
                metrics, _ = self.run_single_attack(
                    original_path, watermarked_path, 'bm3d', sigma, output_dir
                )
                results[f"bm3d_sigma{sigma}"] = metrics

        # VAE compression if available
        if COMPRESSAI_AVAILABLE:
            print("\n=== Running VAE Compression Attacks ===")
            models = ['bmshj2018-factorized', 'mbt2018', 'cheng2020-anchor']
            qualities = [1, 3, 6]
            for model in models:
                for quality in qualities:
                    print(f"\nRunning VAE attack (model={model}, quality={quality})")
                    metrics, _ = self.run_single_attack(
                        original_path, watermarked_path, 'vae', (model, quality), output_dir
                    )
                    results[f"vae_{model}_q{quality}"] = metrics

        # Create summary report
        if output_dir:
            self.create_comprehensive_report(results, output_dir)

        return results

    def create_comprehensive_report(self, results: Dict[str, Dict[str, float]], output_dir: str):
        """
        Create a comprehensive report of all attack results

        Args:
            results: Dictionary of metrics keyed by attack name
            output_dir: Directory to save the report
        """
        with open(os.path.join(output_dir, "comprehensive_attack_report.txt"), 'w') as f:
            f.write("COMPREHENSIVE WATERMARK ATTACK EVALUATION\n")
            f.write("=" * 80 + "\n\n")

            # Group results by attack type
            attack_groups = {}
            for attack_name, metrics in results.items():
                if metrics is None:
                    continue

                # Extract the attack type (before the first underscore)
                attack_type = attack_name.split('_')[0]
                if attack_type not in attack_groups:
                    attack_groups[attack_type] = []
                attack_groups[attack_type].append((attack_name, metrics))

            # Write results by attack type
            for attack_type, attack_results in attack_groups.items():
                f.write(f"\n{attack_type.upper()} ATTACKS\n")
                f.write("-" * 60 + "\n")

                for attack_name, metrics in attack_results:
                    f.write(f"Attack: {attack_name}\n")
                    f.write(f"  PSNR: {metrics['psnr_attack']:.2f} dB\n")
                    f.write(f"  SSIM: {metrics['ssim_attack']:.4f}\n")
                    f.write(f"  LBP Similarity: {metrics['lbp_similarity_attack']:.4f}\n")
                    f.write(f"  GLCM Similarity: {metrics['glcm_similarity_attack']:.4f}\n\n")

            # Sort all attacks by SSIM to find most and least effective
            sorted_attacks = sorted([(k, v) for k, v in results.items() if v is not None],
                                    key=lambda x: x[1]['ssim_attack'])

            f.write("\nTOP 5 MOST EFFECTIVE ATTACKS (lowest SSIM)\n")
            f.write("-" * 60 + "\n")
            for attack_name, metrics in sorted_attacks[:5]:
                f.write(f"{attack_name}: SSIM = {metrics['ssim_attack']:.4f}, PSNR = {metrics['psnr_attack']:.2f} dB\n")

            f.write("\nTOP 5 LEAST EFFECTIVE ATTACKS (highest SSIM)\n")
            f.write("-" * 60 + "\n")
            for attack_name, metrics in sorted_attacks[-5:]:
                f.write(f"{attack_name}: SSIM = {metrics['ssim_attack']:.4f}, PSNR = {metrics['psnr_attack']:.2f} dB\n")

            # Add overall statistics
            if results:
                avg_psnr = np.mean([m['psnr_attack'] for _, m in sorted_attacks])
                avg_ssim = np.mean([m['ssim_attack'] for _, m in sorted_attacks])
                avg_lbp = np.mean([m['lbp_similarity_attack'] for _, m in sorted_attacks])
                avg_glcm = np.mean([m['glcm_similarity_attack'] for _, m in sorted_attacks])

                f.write("\nOVERALL ATTACK STATISTICS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Average PSNR across all attacks: {avg_psnr:.2f} dB\n")
                f.write(f"Average SSIM across all attacks: {avg_ssim:.4f}\n")
                f.write(f"Average LBP Similarity across all attacks: {avg_lbp:.4f}\n")
                f.write(f"Average GLCM Similarity across all attacks: {avg_glcm:.4f}\n")


def add_resd_attack_capability(attack_tool):
    """
    Add the direct ReSD attack capability to an AdvancedWatermarkAttacks instance
    """
    # First make sure run_single_attack method can handle 'diffusion_resd_direct' attack type
    # No need to modify if it already handles this attack type correctly

    # For other attackers that may need a custom method implementation, add it:
    # attack_tool.diffusion_resd_direct_attack = your_implementation

    print("ReSD attack capability added.")
    return attack_tool


# Example of how to use DirectReSDAttack directly without integration
def run_direct_resd_attack(original_path, watermarked_path, output_path, prompt=None, strength=0.3, noise_step=15):
    """
    Run ReSD attack directly without integration with AdvancedWatermarkAttacks

    Args:
        original_path: Path to original image
        watermarked_path: Path to watermarked image
        output_path: Path to save output
        prompt: Text prompt for generation (optional)
        strength: Strength of attack (0-1)
        noise_step: Noise step for attack

    Returns:
        Path to attacked image
    """
    # Initialize the ReSD attacker
    resd_attacker = DirectReSDAttack()

    # Run the attack
    if prompt is None:
        prompt = "A clean, detailed, high-quality photograph"

    attacked_image = resd_attacker.attack_image(
        watermarked_path,
        output_path,
        prompt=prompt,
        noise_step=noise_step,
        strength=strength
    )

    if attacked_image is not None:
        print(f"Attack successful. Saved to {output_path}")
    else:
        print("Attack failed.")

    return output_path

    # Comprehensive evaluation with all attack types


# Example usage with integration into AdvancedWatermarkAttacks
def integrate_with_advanced_attacks(attack_tool):
    """
    Add all watermark attack methods to an existing AdvancedWatermarkAttacks instance

    Args:
        attack_tool: Instance of AdvancedWatermarkAttacks
    """
    # Add all the new attack methods
    attack_tool.gaussian_blur_attack = IntegratedWatermarkAttackers.gaussian_blur_attack.__get__(attack_tool)
    attack_tool.gaussian_noise_attack = IntegratedWatermarkAttackers.gaussian_noise_attack.__get__(attack_tool)
    attack_tool.jpeg_compression_attack = IntegratedWatermarkAttackers.jpeg_compression_attack.__get__(attack_tool)
    attack_tool.brightness_attack = IntegratedWatermarkAttackers.brightness_attack.__get__(attack_tool)
    attack_tool.contrast_attack = IntegratedWatermarkAttackers.contrast_attack.__get__(attack_tool)
    attack_tool.rotation_attack = IntegratedWatermarkAttackers.rotation_attack.__get__(attack_tool)
    attack_tool.scale_attack = IntegratedWatermarkAttackers.scale_attack.__get__(attack_tool)
    attack_tool.crop_attack = IntegratedWatermarkAttackers.crop_attack.__get__(attack_tool)

    # Add advanced attacks if libraries available
    if BM3D_AVAILABLE:
        attack_tool.bm3d_attack = IntegratedWatermarkAttackers.bm3d_attack.__get__(attack_tool)

    if COMPRESSAI_AVAILABLE:
        attack_tool.vae_attack = IntegratedWatermarkAttackers.vae_attack.__get__(attack_tool)

    # Replace run_single_attack with the enhanced version
    attack_tool.run_single_attack = IntegratedWatermarkAttackers.run_single_attack.__get__(attack_tool)

    # Add comprehensive evaluation
    attack_tool.run_comprehensive_evaluation = IntegratedWatermarkAttackers.run_comprehensive_evaluation.__get__(
        attack_tool)
    attack_tool.create_comprehensive_report = IntegratedWatermarkAttackers.create_comprehensive_report.__get__(
        attack_tool)

    return attack_tool


# Example usage script
if __name__ == "__main__":
    from attack_class import AdvancedWatermarkAttacks

    # Define paths
    original_path = 'C:/Users/User/Desktop/stable_signature/output/imgs/000_train_orig.png'
    watermarked_path = 'C:/Users/User/Desktop/stable_signature/output/imgs/000_train_w.png'
    output_dir = 'C:/Users/User/Desktop/stable_signature/output/attack_results'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the attack tool
    attack_tool = AdvancedWatermarkAttacks(device='cpu')  # Use 'cuda' if available

    # Integrate all the new attack methods
    attack_tool = integrate_with_advanced_attacks(attack_tool)

    # Run example of a new attack
    print("Running JPEG compression attack...")
    metrics, attacked_image = attack_tool.run_single_attack(
        original_path,
        watermarked_path,
        attack_type='jpeg',
        param=50,  # JPEG quality
        output_dir=output_dir
    )

    print("Running brightness attack...")
    metrics, attacked_image = attack_tool.run_single_attack(
        original_path,
        watermarked_path,
        attack_type='brightness',
        param=0.8,
        output_dir=output_dir
    )

    # Run direct ReSD attack (from previous implementation)
    print("Running direct ReSD attack...")
    metrics, attacked_image = attack_tool.run_single_attack(
        original_path,
        watermarked_path,
        attack_type='diffusion_resd_direct',
        param=(0.3, 15),  # (strength, noise_step)
        output_dir=output_dir,
        prompt="A detailed photograph of a baby stroller with pink items and a price tag"
    )

    # Run comprehensive evaluation with all methods
    # Uncomment to run a full evaluation (will take significant time)
    # print("\nRunning comprehensive evaluation...")
    # results = attack_tool.run_comprehensive_evaluation(
    #     original_path,
    #     watermarked_path,
    #     output_dir=output_dir
    # )

    print(f"Attacks completed. Results saved to {output_dir}")
