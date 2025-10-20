import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Import the custom ReSDPipeline
# You'll need to ensure this is available in your path
from res_pipe import ReSDPipeline
from integrated_watermark_attackers import integrate_with_advanced_attacks  # noqa: F401

class DirectReSDAttack:
    """
    Implementation of watermark attack using the ReSDPipeline directly.
    This uses the specific capabilities of ReSD for better watermark removal.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 model_path="runwayml/stable-diffusion-v1-5",
                 batch_size=1):
        """
        Initialize the ReSD attack pipeline

        Args:
            device: Device to use (cuda or cpu)
            model_path: Path to pretrained model
            batch_size: Batch size for processing multiple images
        """
        self.device = device
        self.BATCH_SIZE = batch_size

        print(f"Initializing ReSD pipeline on {device}...")
        try:
            # Load the ReSDPipeline model directly
            self.pipe = ReSDPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32
            ).to(device)

            # Set a default noise step
            self.noise_step = 20
            print(f"ReSD pipeline initialized with default noise step {self.noise_step}")
        except Exception as e:
            print(f"Error initializing ReSD pipeline: {str(e)}")
            self.pipe = None

    def attack_image(self, img_path, out_path, prompt="", noise_step=None, strength=0.5, return_latents=False):
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
            print("ReSD pipeline not initialized successfully. Cannot attack.")
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
            print(f"Error attacking image {img_path}: {str(e)}")
            return None

    def attack_batch(self, image_paths, out_paths, prompts=None, noise_step=None, strength=0.5):
        """
        Attack a batch of images

        Args:
            image_paths: List of input image paths
            out_paths: List of output paths
            prompts: List of prompts (or None for default)
            noise_step: Noise step to use
            strength: Strength parameter

        Returns:
            List of attacked images
        """
        if self.pipe is None:
            print("ReSD pipeline not initialized successfully. Cannot attack.")
            return None

        # Use provided noise step or default
        noise_step = noise_step if noise_step is not None else self.noise_step

        # Use default prompts if none provided
        if prompts is None:
            prompts = ["A detailed high-quality photograph"] * len(image_paths)
        elif isinstance(prompts, str):
            prompts = [prompts] * len(image_paths)

        results = []

        # Process in batches
        for i in range(0, len(image_paths), self.BATCH_SIZE):
            batch_imgs = image_paths[i:i + self.BATCH_SIZE]
            batch_outs = out_paths[i:i + self.BATCH_SIZE]
            batch_prompts = prompts[i:i + self.BATCH_SIZE]

            print(f"Processing batch {i // self.BATCH_SIZE + 1}/{(len(image_paths) - 1) // self.BATCH_SIZE + 1}")

            # Process each image in the batch
            for img_path, out_path, prompt in zip(batch_imgs, batch_outs, batch_prompts):
                result = self.attack_image(
                    img_path,
                    out_path,
                    prompt=prompt,
                    noise_step=noise_step,
                    strength=strength
                )
                results.append(result)

        return results


# Integration with AdvancedWatermarkAttacks class
def integrate_with_advanced_attacks(attack_tool):
    """
    Add ReSD attack capabilities to an existing AdvancedWatermarkAttacks instance

    Args:
        attack_tool: Instance of AdvancedWatermarkAttacks
    """
    # Create a ReSD attacker
    resd_attacker = DirectReSDAttack(device=attack_tool.device)

    # Store it in the attack tool instance
    attack_tool.resd_attacker = resd_attacker

    # Add method to run ReSD attack
    def diffusion_resd_direct_attack(self, img, prompt="A detailed high-quality photograph", noise_step=20,
                                     strength=0.5):
        """
        Run ReSD attack using the direct integration
        Note: This requires the image to be saved temporarily
        """
        import tempfile
        import os

        if not hasattr(self, 'resd_attacker') or self.resd_attacker is None:
            print("ReSD attacker not initialized. Cannot perform attack.")
            return img

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_in, \
                tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_out:
            # Save input image to temporary file
            input_path = tmp_in.name
            output_path = tmp_out.name

        # Close temp files so they can be accessed on Windows
        cv2.imwrite(input_path, img)

        # Run attack
        try:
            result = self.resd_attacker.attack_image(
                input_path,
                output_path,
                prompt=prompt,
                noise_step=noise_step,
                strength=strength
            )

            # If attack successful, read the result
            if result is not None:
                attacked_img = result
            else:
                # Read from saved output if direct return failed
                attacked_img = cv2.imread(output_path)

            # Clean up temp files
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass

            return attacked_img

        except Exception as e:
            print(f"Error in ReSD direct attack: {str(e)}")
            print("Falling back to original image")

            # Clean up temp files
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass

            return img

    # Add the new method to the attack tool
    attack_tool.diffusion_resd_direct_attack = diffusion_resd_direct_attack.__get__(attack_tool)

    # Extend run_single_attack to include the new attack type
    original_run_single_attack = attack_tool.run_single_attack

    def extended_run_single_attack(self, original_path, watermarked_path, attack_type, param=None, output_dir=None,
                                   prompt=None):
        """Extended run_single_attack with direct ReSD support"""
        if attack_type == 'diffusion_resd_direct':
            # Parse parameters
            strength = 0.5
            noise_step = 20

            if param is not None:
                if isinstance(param, tuple) and len(param) == 2:
                    strength, noise_step = param
                else:
                    strength = param

            # Use default prompt if none provided
            if prompt is None:
                prompt = "A detailed high-quality photograph"

            print(f"Running direct ReSD attack (strength={strength}, noise_step={noise_step}, prompt='{prompt}')")

            # Load images
            original_image = cv2.imread(original_path)
            watermarked_image = cv2.imread(watermarked_path)

            if original_image is None or watermarked_image is None:
                print("Error: Could not load one or both images. Please check the file paths.")
                return None

            # Run the attack
            attacked_image = self.diffusion_resd_direct_attack(
                watermarked_image,
                prompt=prompt,
                noise_step=noise_step,
                strength=strength
            )

            if attacked_image is None:
                print("Attack failed.")
                return None

            # Parameter string for output filename
            param_str = f"strength={strength},noise={noise_step}"

            # Evaluate and visualize results
            metrics = self.plot_attack_comparison(
                original_image, watermarked_image, attacked_image,
                attack_type, param_str, output_dir
            )

            # Save attacked image if output directory is provided
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f"attacked_{attack_type}_{param_str}.png"), attacked_image)

            return metrics, attacked_image
        else:
            # Use the original method for other attack types
            return original_run_single_attack(self, original_path, watermarked_path, attack_type, param, output_dir)

    # Replace the method
    attack_tool.run_single_attack = extended_run_single_attack.__get__(attack_tool)

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
    attack_tool = AdvancedWatermarkAttacks(device='cuda')
    # Integrate ReSD attack
    attack_tool = integrate_with_advanced_attacks(attack_tool)

    # Run direct ReSD attack
    print("Running direct ReSD attack...")
    # metrics, attacked_image = attack_tool.run_single_attack(
    #     original_path,
    #     watermarked_path,
    #     attack_type='diffusion_resd_direct',
    #     param=(0.3, 15),  # (strength, noise_step)
    #     output_dir=output_dir,
    #     prompt="A detailed photograph of a baby stroller with pink items and a price tag"
    # )
    #
    # print(f"Attack completed. Results saved to {output_dir}")
    results = attack_tool.run_comprehensive_evaluation(
        original_path,
        watermarked_path,
        output_dir=output_dir
    )