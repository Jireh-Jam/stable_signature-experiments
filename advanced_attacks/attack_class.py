import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import torch
from PIL import Image
from torchvision import transforms
import os
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Optional, Tuple, Dict, Any
from common.logging_utils import get_logger

logger = get_logger(__name__)

# For diffusion models
try:
    from diffusers import StableDiffusionInpaintPipeline,StableDiffusionPipeline, DiffusionPipeline, StableDiffusionImg2ImgPipeline, \
        AutoPipelineForImage2Image
    import torch

    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    print("Diffusion models not available. Install with: pip install diffusers transformers accelerate")

# For adversarial attacks
try:
    import foolbox as fb
    import torchvision.models as models

    ADVERSARIAL_AVAILABLE = True
except ImportError:
    ADVERSARIAL_AVAILABLE = False
    print("Adversarial attack tools not available. Install with: pip install foolbox")


class AdvancedWatermarkAttacks:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        ])

        # Initialize models if available
        if DIFFUSION_AVAILABLE:
            logger.info(f"Loading diffusion models on {self.device}...")
            # Initialize Stable Diffusion for inpainting
            self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)

            # Initialize Stable Diffusion for img2img
            self.img2img_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)

            # Initialize AutoPipeline for img2img
            self.img2img_model = AutoPipelineForImage2Image.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)

        if ADVERSARIAL_AVAILABLE:
            logger.info("Loading adversarial attack models...")
            # Load a pre-trained model for adversarial attacks
            self.model = models.resnet50(pretrained=True).to(self.device).eval()
            # Create Foolbox model
            self.fmodel = fb.PyTorchModel(self.model, bounds=(0, 1))


    def diffusion_resd_attack(self, img, prompt="A detailed, high-quality photograph", noise_step=20, strength=0.5):
        """
        Attack using ReSD approach (Regeneration Stable Diffusion) to remove watermarks

        Args:
            img: Input watermarked image (OpenCV BGR format)
            prompt: Text prompt to guide the diffusion model (general by default)
            noise_step: Number of noise steps to add (higher = more modification)
            strength: Strength of diffusion effect (0-1)

        Returns:
            Processed image with potentially removed watermark
        """
        if not DIFFUSION_AVAILABLE:
            logger.warning("Diffusion models not available. Skipping ReSD attack.")
            return img

        # Store original dimensions
        original_h, original_w = img.shape[:2]

        # Convert from OpenCV BGR to RGB PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Resize to valid dimensions if needed (multiple of 8)
        width, height = img_pil.size
        if width % 8 != 0 or height % 8 != 0:
            width = (width // 8) * 8
            height = (height // 8) * 8
            img_pil = img_pil.resize((width, height))

        try:
            # Run img2img with the general prompt
            with torch.no_grad():
                outputs = self.img2img_model(
                    prompt=prompt,
                    image=img_pil,
                    strength=strength,
                    guidance_scale=7.5,
                    num_inference_steps=30
                )

            output_img = outputs.images[0]

            # Convert back to OpenCV format
            output_cv = cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)

            # Resize back to original dimensions if necessary
            if output_cv.shape[0] != original_h or output_cv.shape[1] != original_w:
                output_cv = cv2.resize(output_cv, (original_w, original_h), interpolation=cv2.INTER_AREA)

            return output_cv

        except Exception as e:
            logger.warning(f"Error in diffusion ReSD attack: {e}. Falling back to original image")
            return img

    def diffusion_regeneration_attack(self, img, prompt="A clear photograph of a baby stroller or buggy with a pink "
                                                        "item inside and a price tag visible in high definition",
                                      strength=0.7):
        """Attack using stable diffusion img2img to regenerate the image"""
        if not DIFFUSION_AVAILABLE:
            logger.warning("Diffusion models not available. Skipping regeneration attack.")
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
            logger.debug(f"Resizing output from {current_w}x{current_h} to {original_w}x{original_h}")
            output_cv = cv2.resize(output_cv, (original_w, original_h), interpolation=cv2.INTER_AREA)

        return output_cv

    def diffusion_image_to_image_attack(self, img, prompt="A normal image", strength=0.7):
        """Attack using stable diffusion img2img to regenerate the image"""
        if not DIFFUSION_AVAILABLE:
            logger.warning("Diffusion models not available. Skipping regeneration attack.")
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
            logger.debug(f"Resizing output from {current_w}x{current_h} to {original_w}x{original_h}")
            output_cv = cv2.resize(output_cv, (original_w, original_h), interpolation=cv2.INTER_AREA)

        return output_cv

    def diffusion_inpainting_attack(self, img, prompt="A clear photograph of a baby stroller or buggy with a pink "
                                                        "item inside and a price tag visible in high definition", mask_ratio=0.3, strength=0.75):
        """Attack using stable diffusion inpainting"""
        if not DIFFUSION_AVAILABLE:
            logger.warning("Diffusion models not available. Skipping inpainting attack.")
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
            logger.debug(f"Resizing output from {current_w}x{current_h} to {original_w}x{original_h}")
            output_cv = cv2.resize(output_cv, (original_w, original_h), interpolation=cv2.INTER_AREA)

        return output_cv

    def high_frequency_attack(self, img, threshold_percentile=95, filter_strength=0.8):
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

    def analyze_frequency_components(self, img, title="Image Frequency Analysis", output_path=None):
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

        # # Visualize
        # plt.figure(figsize=(15, 5))

        # plt.subplot(1, 3, 1)
        # plt.title('Original Image')
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) > 2 else gray, cmap='gray')
        # plt.axis('off')

        # plt.subplot(1, 3, 2)
        # plt.title('Magnitude Spectrum')
        # plt.imshow(magnitude_spectrum, cmap='viridis')
        # plt.colorbar(label='Log Magnitude')
        # plt.axis('off')

        # plt.subplot(1, 3, 3)
        # plt.title('High Frequency Components')
        # plt.imshow(high_frequency_mask, cmap='gray')
        # plt.axis('off')

        # plt.suptitle(title)
        # plt.tight_layout()

        # if output_path:
        #     plt.savefig(output_path)

        # plt.show()

        return magnitude_spectrum, high_frequency_mask


    def adversarial_attack(self, img, attack_type='FGSM', epsilon=0.03):
        """Generate adversarial examples using various attacks"""
        if not ADVERSARIAL_AVAILABLE:
            logger.warning("Adversarial attack tools not available. Skipping attack.")
            return img
        # Store original dimensions
        original_h, original_w = img.shape[:2]
        # Convert to PyTorch tensor
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # Ensure the input is properly bounded to [0, 1] for Foolbox
        img_tensor_normalized = torch.clamp(img_tensor, 0, 1)

        # Create target (just use a random class for demonstration)
        target = torch.tensor([283]).to(self.device)  # 283 is 'Persian cat' in ImageNet

        # Create attack
        if attack_type == 'FGSM':
            attack = fb.attacks.FGSM()
        elif attack_type == 'PGD':
            attack = fb.attacks.PGD()
        elif attack_type == 'DeepFool':
            attack = fb.attacks.L2DeepFoolAttack()
        else:
            attack = fb.attacks.FGSM()  # Default

        # Print bounds for debugging
        logger.debug(f"Model bounds: {self.fmodel.bounds}")
        logger.debug(f"Input tensor min: {img_tensor_normalized.min().item()}, max: {img_tensor_normalized.max().item()}")

        try:
            # Generate adversarial example
            _, adv_img, _ = attack(self.fmodel, img_tensor_normalized, target, epsilons=epsilon)

            # Convert back to image
            adv_img = self.reverse_transform(adv_img.squeeze(0)).cpu().numpy()
            adv_img = np.clip(adv_img.transpose(1, 2, 0), 0, 1)
            adv_img = (adv_img * 255).astype(np.uint8)

            # Convert to BGR for OpenCV
            adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)

            # Ensure output image has the same dimensions as input
            current_h, current_w = adv_img.shape[:2]
            if current_h != original_h or current_w != original_w:
                print(f"Resizing output from {current_w}x{current_h} to {original_w}x{original_h}")
                adv_img = cv2.resize(adv_img, (original_w, original_h), interpolation=cv2.INTER_AREA)
            return adv_img

        except Exception as e:
            logger.warning(f"Error in adversarial attack: {e}. Falling back to original image")
            return img

    def calculate_texture_features(self, image):
        """Calculate texture features using LBP and GLCM for a single image"""
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # LBP features
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist = np.histogram(lbp, bins=59, range=(0, 59))[0]
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
        glcm = graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)

        # Properties to extract
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

        # Create dictionaries to store features by property name
        glcm_features = {}

        for prop in properties:
            glcm_features[prop] = graycoprops(glcm, prop).flatten()

        return {
            'lbp_hist': lbp_hist,
            'lbp_stats': lbp_stats,
            'img_stats': img_stats,
            'glcm_features': glcm_features
        }

    def calculate_similarity_metrics(self, features_orig, features_attacked):
        """Calculate similarity metrics between original and attacked image features"""
        # LBP histogram similarity
        lbp_similarity = 1 - np.sum(np.abs(features_orig['lbp_hist'] - features_attacked['lbp_hist'])) / 2

        # GLCM similarities
        glcm_similarities = {}
        for prop in features_orig['glcm_features'].keys():
            orig = features_orig['glcm_features'][prop]
            attacked = features_attacked['glcm_features'][prop]
            # Cosine similarity
            similarity = np.dot(orig, attacked) / (np.linalg.norm(orig) * np.linalg.norm(attacked))
            glcm_similarities[prop] = similarity

        return {
            'lbp_similarity': lbp_similarity,
            'glcm_similarities': glcm_similarities
        }

    def calculate_image_metrics(self, orig_image, attacked_image):
        """Calculate PSNR and SSIM between two images"""
        # Calculate PSNR
        mse = np.mean((orig_image.astype(np.float64) - attacked_image.astype(np.float64)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255 / np.sqrt(mse))

        # Calculate SSIM (on grayscale images)
        orig_gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        attacked_gray = cv2.cvtColor(attacked_image, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(orig_gray, attacked_gray)

        return psnr, ssim_value

    def plot_attack_comparison(self, orig_image, watermarked_image, attacked_image, attack_type, param=None,
                               output_path=None):
        """Plot original, watermarked, and attacked images along with metrics"""
        # Calculate metrics
        original_features = self.calculate_texture_features(orig_image)
        watermarked_features = self.calculate_texture_features(watermarked_image)
        attacked_features = self.calculate_texture_features(attacked_image)

        # Calculate similarity metrics
        watermark_metrics = self.calculate_similarity_metrics(original_features, watermarked_features)
        attack_metrics = self.calculate_similarity_metrics(watermarked_features, attacked_features)

        # Calculate PSNR and SSIM
        psnr_watermark, ssim_watermark = self.calculate_image_metrics(orig_image, watermarked_image)
        psnr_attack, ssim_attack = self.calculate_image_metrics(watermarked_image, attacked_image)

        # Create figure for images
        fig = plt.figure(figsize=(15, 10))

        # Plot images
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB))
        plt.title('Watermarked Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(attacked_image, cv2.COLOR_BGR2RGB))
        if param is not None:
            plt.title(f'Attacked Image ({attack_type}, param={param})')
        else:
            plt.title(f'Attacked Image ({attack_type})')
        plt.axis('off')

        # Add metrics text boxes
        plt.figtext(0.3, 0.05, f"Original vs Watermarked:\nPSNR: {psnr_watermark:.2f} dB\nSSIM: {ssim_watermark:.4f}",
                    bbox=dict(facecolor='white', alpha=0.8), horizontalalignment='center')

        plt.figtext(0.7, 0.05, f"Watermarked vs Attacked:\nPSNR: {psnr_attack:.2f} dB\nSSIM: {ssim_attack:.4f}",
                    bbox=dict(facecolor='white', alpha=0.8), horizontalalignment='center')

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        if output_path:
            plt.savefig(os.path.join(output_path, f"img_comparison_{attack_type}_{param}.png"))

        plt.show()

        # Create new figure for feature analysis
        fig = plt.figure(figsize=(15, 10))

        # LBP histogram
        plt.subplot(2, 1, 1)
        x = np.arange(len(original_features['lbp_hist']))
        width = 0.25
        plt.bar(x - width, original_features['lbp_hist'], width, label='Original', alpha=0.7)
        plt.bar(x, watermarked_features['lbp_hist'], width, label='Watermarked', alpha=0.7)
        plt.bar(x + width, attacked_features['lbp_hist'], width, label='Attacked', alpha=0.7)
        plt.title(f'LBP Histogram Comparison')
        plt.xlabel('LBP Pattern')
        plt.ylabel('Normalized Frequency')

        # Add LBP similarity text box
        plt.figtext(0.5, 0.55, f"LBP Similarity - Orig vs Watermarked: {watermark_metrics['lbp_similarity']:.4f}\n"
                               f"LBP Similarity - Watermarked vs Attacked: {attack_metrics['lbp_similarity']:.4f}",
                    bbox=dict(facecolor='white', alpha=0.8), horizontalalignment='center')

        plt.legend()

        # GLCM properties
        plt.subplot(2, 1, 2)
        properties = list(original_features['glcm_features'].keys())
        x = np.arange(len(properties))
        width = 0.25

        # For each property, average the values across angles for simplicity
        original_means = [np.mean(original_features['glcm_features'][prop]) for prop in properties]
        watermarked_means = [np.mean(watermarked_features['glcm_features'][prop]) for prop in properties]
        attacked_means = [np.mean(attacked_features['glcm_features'][prop]) for prop in properties]

        plt.bar(x - width, original_means, width, label='Original', alpha=0.7)
        plt.bar(x, watermarked_means, width, label='Watermarked', alpha=0.7)
        plt.bar(x + width, attacked_means, width, label='Attacked', alpha=0.7)

        plt.ylabel('GLCM Property Value')
        plt.title('GLCM Properties Comparison')
        plt.xticks(x, properties)

        # Add GLCM similarity text box
        glcm_watermark_avg = np.mean([sim for sim in watermark_metrics['glcm_similarities'].values()])
        glcm_attack_avg = np.mean([sim for sim in attack_metrics['glcm_similarities'].values()])

        plt.figtext(0.5, 0.15, f"Avg GLCM Similarity - Orig vs Watermarked: {glcm_watermark_avg:.4f}\n"
                               f"Avg GLCM Similarity - Watermarked vs Attacked: {glcm_attack_avg:.4f}",
                    bbox=dict(facecolor='white', alpha=0.8), horizontalalignment='center')

        plt.legend()

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        if output_path:
            plt.savefig(os.path.join(output_path, f"feature_comparison_{attack_type}_{param}.png"))

        plt.show()

        # Print metrics
        print(f"\n=== {attack_type.upper()} ATTACK {param if param else ''} ===")
        print(f"Original vs Watermarked: PSNR = {psnr_watermark:.2f} dB, SSIM = {ssim_watermark:.4f}")
        print(f"Watermarked vs Attacked: PSNR = {psnr_attack:.2f} dB, SSIM = {ssim_attack:.4f}")
        print(f"LBP Similarity - Orig vs Watermarked: {watermark_metrics['lbp_similarity']:.4f}")
        print(f"LBP Similarity - Watermarked vs Attacked: {attack_metrics['lbp_similarity']:.4f}")
        print(f"Avg GLCM Similarity - Orig vs Watermarked: {glcm_watermark_avg:.4f}")
        print(f"Avg GLCM Similarity - Watermarked vs Attacked: {glcm_attack_avg:.4f}")

        # Return metrics for reporting
        return {
            'psnr_watermark': psnr_watermark,
            'ssim_watermark': ssim_watermark,
            'psnr_attack': psnr_attack,
            'ssim_attack': ssim_attack,
            'lbp_similarity_watermark': watermark_metrics['lbp_similarity'],
            'lbp_similarity_attack': attack_metrics['lbp_similarity'],
            'glcm_similarity_watermark': glcm_watermark_avg,
            'glcm_similarity_attack': glcm_attack_avg
        }

    def run_evaluation(self, original_path, watermarked_path, output_dir=None):
        """Run evaluation with all available attack methods including high frequency attacks"""
        # Load images
        original_image = cv2.imread(original_path)
        watermarked_image = cv2.imread(watermarked_path)

        if original_image is None or watermarked_image is None:
            logger.error("Could not load one or both images. Please check the file paths.")
            return

        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = {}

        # Analyze frequency components of the watermarked image
        if output_dir:
            freq_analysis_path = os.path.join(output_dir, "frequency_analysis.png")
            self.analyze_frequency_components(
                watermarked_image,
                title="Frequency Analysis of Watermarked Image",
                output_path=freq_analysis_path
            )

        # Run high frequency attacks with different parameters
        logger.info("Running High Frequency Attacks")
        thresholds = [75, 90, 95, 98]
        strengths = [0.5, 0.5, 0.8, 0.95]

        for threshold in thresholds:
            for strength in strengths:
                attack_name = f"high_frequency"
                param_str = f"thresh={threshold},str={strength}"
                logger.info(f"High frequency attack ({param_str})")

                attacked_image = self.high_frequency_attack(
                    watermarked_image,
                    threshold_percentile=threshold,
                    filter_strength=strength
                )

                # Save the attacked image if output dir specified
                if output_dir:
                    cv2.imwrite(
                        os.path.join(output_dir, f"high_freq_thresh{threshold}_str{strength}.png"),
                        attacked_image
                    )

                # Evaluate and compare
                metrics = self.plot_attack_comparison(
                    original_image,
                    watermarked_image,
                    attacked_image,
                    attack_name,
                    param_str,
                    output_dir
                )

                # Add to results
                results[f"high_frequency_thresh{threshold}_str{strength}"] = metrics

        # Diffusion inpainting attacks
        if DIFFUSION_AVAILABLE:
            mask_ratios = [0.2, 0.4, 0.6]
            #prompts = ["A normal image", "Photo of a scene"]
            prompts = ["A normal image", "Be creative", "Photo of a scene"]

            for mask_ratio in mask_ratios:
                for prompt in prompts:
                    logger.info(f"Diffusion inpainting (mask={mask_ratio}, prompt='{prompt}')")
                    attacked_image = self.diffusion_inpainting_attack(
                        watermarked_image, prompt=prompt, mask_ratio=mask_ratio
                    )
                    metrics = self.plot_attack_comparison(
                        original_image, watermarked_image, attacked_image,
                        f"diffusion_inpaint", f"mask={mask_ratio},prompt={prompt}",
                        output_dir
                    )
                    results[f"diffusion_inpaint_mask{mask_ratio}_prompt{prompt}"] = metrics

            # Diffusion regeneration attacks
            strengths = [0.3, 0.5, 0.7]
            for strength in strengths:
                logger.info(f"Diffusion regeneration (strength={strength})")
                attacked_image = self.diffusion_regeneration_attack(
                    watermarked_image, prompt="A normal image", strength=strength
                )
                metrics = self.plot_attack_comparison(
                    original_image, watermarked_image, attacked_image,
                    f"diffusion_regeneration", f"strength={strength}",
                    output_dir
                )
                results[f"diffusion_regeneration_strength{strength}"] = metrics
            # Diffusion image to image attacks
            strengths = [0.3, 0.5, 0.7]
            for strength in strengths:
                logger.info(f"Diffusion image-to-image (strength={strength})")
                attacked_image = self.diffusion_image_to_image_attack(
                    watermarked_image, prompt="A normal image", strength=strength
                )
                metrics = self.plot_attack_comparison(
                    original_image, watermarked_image, attacked_image,
                    f"diffusion_image_to_image", f"strength={strength}",
                    output_dir
                )
                results[f"diffusion_image_to_image_strength{strength}"] = metrics

        # Adversarial attacks
        if ADVERSARIAL_AVAILABLE:
            attack_types = ['FGSM', 'PGD', 'DeepFool']
            epsilons = [0.01, 0.03, 0.05]

            for attack_type in attack_types:
                for epsilon in epsilons:
                    logger.info(f"Adversarial {attack_type} (epsilon={epsilon})")
                    attacked_image = self.adversarial_attack(
                        watermarked_image, attack_type=attack_type, epsilon=epsilon
                    )
                    metrics = self.plot_attack_comparison(
                        original_image, watermarked_image, attacked_image,
                        f"adversarial_{attack_type}", f"epsilon={epsilon}",
                        output_dir
                    )
                    results[f"adversarial_{attack_type}_epsilon{epsilon}"] = metrics

        # Create a summary report
        if output_dir:
            self.create_summary_report(results, output_dir)

        return results

    def create_summary_report(self, results, output_dir):
        """Create a summary report of all attack results"""
        with open(os.path.join(output_dir, "attack_summary_report.txt"), 'w') as f:
            f.write("WATERMARK ROBUSTNESS EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            for attack_name, metrics in results.items():
                f.write(f"Attack: {attack_name}\n")
                f.write("-" * 60 + "\n")
                f.write(f"PSNR (Watermarked vs Attacked): {metrics['psnr_attack']:.2f} dB\n")
                f.write(f"SSIM (Watermarked vs Attacked): {metrics['ssim_attack']:.4f}\n")
                f.write(f"LBP Similarity: {metrics['lbp_similarity_attack']:.4f}\n")
                f.write(f"GLCM Similarity: {metrics['glcm_similarity_attack']:.4f}\n\n")

            # Sort attacks by SSIM to find most and least effective
            sorted_attacks = sorted(results.items(), key=lambda x: x[1]['ssim_attack'])

            f.write("TOP 3 MOST EFFECTIVE ATTACKS (lowest SSIM)\n")
            f.write("-" * 60 + "\n")
            for attack_name, metrics in sorted_attacks[:3]:
                f.write(f"{attack_name}: SSIM = {metrics['ssim_attack']:.4f}, PSNR = {metrics['psnr_attack']:.2f} dB\n")

            f.write("\nTOP 3 LEAST EFFECTIVE ATTACKS (highest SSIM)\n")
            f.write("-" * 60 + "\n")
            for attack_name, metrics in sorted_attacks[-3:]:
                f.write(f"{attack_name}: SSIM = {metrics['ssim_attack']:.4f}, PSNR = {metrics['psnr_attack']:.2f} dB\n")

    def run_single_attack(self, original_path, watermarked_path, attack_type, param=None, output_dir=None, prompt=None):
        """
        Run a single attack for quick testing

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
            logger.error("Could not load one or both images. Please check the file paths.")
            return

        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Apply the selected attack
        if attack_type == 'diffusion_inpainting':
            mask_ratio = param if param is not None else 0.3
            logger.info(f"Diffusion inpainting (mask ratio={mask_ratio})")
            attacked_image = self.diffusion_inpainting_attack(
                watermarked_image, prompt=prompt or "A clean, high-quality photograph", mask_ratio=mask_ratio
            )
            param_str = f"mask={mask_ratio}"

        elif attack_type == 'diffusion_regeneration':
            strength = param if param is not None else 0.5
            logger.info(f"Diffusion regeneration (strength={strength})")
            attacked_image = self.diffusion_regeneration_attack(
                watermarked_image, prompt=prompt or "A clean, high-quality photograph", strength=strength
            )
            param_str = f"strength={strength}"

        elif attack_type == 'diffusion_resd':
            strength = param if param is not None else 0.5
            noise_step = 20  # Default noise step

            if isinstance(param, tuple) and len(param) == 2:
                strength, noise_step = param

            logger.info(f"Diffusion ReSD (strength={strength}, noise_step={noise_step})")

            # Use general prompt if not specified
            default_prompt = "A clean, detailed, high-quality photograph"

            attacked_image = self.diffusion_resd_attack(
                watermarked_image,
                prompt=prompt or default_prompt,
                noise_step=noise_step,
                strength=strength
            )
            param_str = f"strength={strength},noise={noise_step}"

        elif attack_type == 'diffusion_image_to_image':
            strength = param if param is not None else 0.5
            logger.info(f"Diffusion image-to-image (strength={strength})")
            attacked_image = self.diffusion_image_to_image_attack(
                watermarked_image, prompt=prompt or "A clean, high-quality photograph", strength=strength
            )
            param_str = f"strength={strength}"

        elif attack_type.startswith('adversarial_'):
            adv_type = attack_type.split('_')[1] if '_' in attack_type else 'FGSM'
            epsilon = param if param is not None else 0.03
            logger.info(f"Adversarial {adv_type} (epsilon={epsilon})")
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

            logger.info(f"High frequency (threshold={threshold}, strength={strength})")

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
            logger.error(f"Unknown attack type: {attack_type}")
            return

        # Evaluate and visualize results
        metrics = self.plot_attack_comparison(
            original_image, watermarked_image, attacked_image,
            attack_type, param_str, output_dir
        )

        # Save attacked image if output directory is provided
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"attacked_{attack_type}_{param_str}.png"), attacked_image)

        return metrics, attacked_image


def main():
    """Main function for command line usage"""
    import argparse
    parser = argparse.ArgumentParser(description='Advanced Watermark Attacks')
    parser.add_argument('--original', required=True, help='Path to original image')
    parser.add_argument('--watermarked', required=True, help='Path to watermarked image')
    parser.add_argument('--output', default='results', help='Output directory for results')
    parser.add_argument('--attack', default='high_frequency',
                        choices=['high_frequency', 'diffusion_inpainting', 'diffusion_regeneration','diffusion_image_to_image',
                                 'adversarial_FGSM', 'adversarial_PGD', 'adversarial_DeepFool', 'all'],
                        help='Attack type to run')
    parser.add_argument('--param', type=float, default=None, help='Attack parameter (threshold, strength, etc.)')
    parser.add_argument('--device', default=None, help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize attack tool
    attack_tool = AdvancedWatermarkAttacks(device=device)

    # Run attacks
    if args.attack == 'all':
        attack_tool.run_evaluation(args.original, args.watermarked, args.output)
    else:
        attack_tool.run_single_attack(args.original, args.watermarked, args.attack, args.param, args.output)


if __name__ == "__main__":
    main()
