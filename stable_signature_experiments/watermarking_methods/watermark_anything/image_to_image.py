import torch
import os
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, EulerDiscreteScheduler
from attribution import MappingNetwork
from customization import customize_vae_decoder
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import numpy as np

def get_phis(phi_dimension, batch_size, eps=1e-8):
    """Generate random fingerprints - matching the training script"""
    phi_length = phi_dimension
    b = batch_size
    phi = torch.empty(b, phi_length).uniform_(0, 1)
    return torch.bernoulli(phi) + eps

def get_custom_fingerprint(fingerprint_bits):
    """Convert a list/array of bits to fingerprint tensor"""
    if isinstance(fingerprint_bits, (list, np.ndarray)):
        fingerprint_bits = torch.tensor(fingerprint_bits, dtype=torch.float32)
    return fingerprint_bits.unsqueeze(0) if fingerprint_bits.dim() == 1 else fingerprint_bits

def embed_watermark_in_image(
    image_path, 
    model_path, 
    output_path=None, 
    custom_fingerprint=None,
    strength=0.7,  # How much to modify the image (0.0 = no change, 1.0 = completely new)
    num_inference_steps=20
):
    """
    Embed a watermark into an existing image
    
    Args:
        image_path: Path to input image
        model_path: Path to WOUAF model
        output_path: Path to save watermarked image
        custom_fingerprint: Optional custom fingerprint (list of 0s and 1s)
        strength: How much to modify the image (0.0-1.0)
        num_inference_steps: Number of denoising steps
    """
    
    print("🔄 Loading WOUAF model...")
    
    # Parameters - should match your training configuration
    phi_dimension = 32
    int_dimension = 128
    lr_mult = 1.0
    mapping_layer = 2
    
    # Load base VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="vae")
    
    # Customize VAE with same parameters as training
    vae = customize_vae_decoder(vae, int_dimension, lr_mult)
    
    # Load trained VAE weights
    vae_weights_path = os.path.join(model_path, "vae_decoder.pth")
    if os.path.exists(vae_weights_path):
        vae_weights = torch.load(vae_weights_path, map_location='cpu')
        vae.decoder.load_state_dict(vae_weights)
        print("✅ VAE decoder weights loaded successfully!")
    else:
        print(f"⚠️ VAE weights not found at {vae_weights_path}")
        return None
    
    # Load mapping network
    mapping_network = MappingNetwork(phi_dimension, int_dimension, num_layers=mapping_layer)
    mapping_weights_path = os.path.join(model_path, "mapping_network.pth")
    if os.path.exists(mapping_weights_path):
        mapping_weights = torch.load(mapping_weights_path, map_location='cpu')
        mapping_network.load_state_dict(mapping_weights)
        mapping_network.eval()
        print("✅ Mapping network loaded successfully!")
    else:
        print(f"⚠️ Mapping network weights not found at {mapping_weights_path}")
        return None
    
    # Load decoding network (for verification)
    decoding_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    decoding_network.fc = torch.nn.Linear(2048, phi_dimension)
    decoding_weights_path = os.path.join(model_path, "decoding_network.pth")
    if os.path.exists(decoding_weights_path):
        decoding_weights = torch.load(decoding_weights_path, map_location='cpu')
        decoding_network.load_state_dict(decoding_weights)
        decoding_network.eval()
        print("✅ Decoding network loaded successfully!")
    else:
        print(f"⚠️ Decoding network weights not found at {decoding_weights_path}")
        decoding_network = None
    
    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = vae.to(device)
    mapping_network = mapping_network.to(device)
    if decoding_network is not None:
        decoding_network = decoding_network.to(device)
    
    print(f"🖼️ Loading input image from {image_path}")
    
    # Load and preprocess the input image
    input_image = Image.open(image_path).convert("RGB")
    
    # Resize to 512x512 (standard Stable Diffusion size)
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    image_tensor = transform(input_image).unsqueeze(0).to(device)
    
    print("🔄 Encoding image to latent space...")
    
    # Encode image to latent space
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = latents * 0.18215  # VAE scaling factor
    
    print("🎯 Generating fingerprint...")
    
    # Generate or use custom fingerprint
    if custom_fingerprint is not None:
        if len(custom_fingerprint) != phi_dimension:
            raise ValueError(f"Custom fingerprint must have {phi_dimension} bits, got {len(custom_fingerprint)}")
        phis = get_custom_fingerprint(custom_fingerprint).to(device)
        print(f"📋 Using custom fingerprint: {custom_fingerprint}")
    else:
        phis = get_phis(phi_dimension, 1).to(device)
        print(f"🎲 Generated random fingerprint: {(phis > 0.5).int().cpu().numpy().flatten()}")
    
    # Encode fingerprint
    with torch.no_grad():
        encoded_fingerprint = mapping_network(phis)
    
    print("💧 Embedding watermark...")
    
    # Add noise to latents based on strength (this simulates the denoising process)
    if strength > 0:
        noise = torch.randn_like(latents)
        # Add noise proportional to strength
        noisy_latents = latents + noise * strength * 0.1  # Scale noise appropriately
    else:
        noisy_latents = latents
    
    # Decode with watermark embedding
    with torch.no_grad():
        watermarked_image = vae.decode(noisy_latents, encoded_fingerprint).sample
    
    # Convert back to PIL image
    watermarked_image = (watermarked_image / 2 + 0.5).clamp(0, 1)
    watermarked_image = watermarked_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
    watermarked_image = (watermarked_image * 255).astype(np.uint8)
    watermarked_pil = Image.fromarray(watermarked_image)
    
    # Save watermarked image
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"./test_results/{base_name}_watermarked.png"
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "./test_results", exist_ok=True)
    watermarked_pil.save(output_path)
    print(f"💾 Watermarked image saved to {output_path}")
    
    # Verify watermark extraction
    if decoding_network is not None:
        print("🔍 Verifying watermark extraction...")
        
        # Convert back to tensor for verification
        verify_transform = transforms.Compose([transforms.ToTensor()])
        verify_tensor = verify_transform(watermarked_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            extracted_keys = decoding_network(verify_tensor)
            extracted_binary = (torch.sigmoid(extracted_keys) > 0.5).int()
            original_binary = (phis > 0.5).int()
            
            # Calculate accuracy
            bit_acc = ((original_binary == extracted_binary).sum(dim=1).float() / phi_dimension).item()
            
            print(f"📊 Original fingerprint:  {original_binary.cpu().numpy().flatten()}")
            print(f"📊 Extracted fingerprint: {extracted_binary.cpu().numpy().flatten()}")
            print(f"📊 Bit accuracy: {bit_acc:.4f}")
    
    print("✅ Watermark embedding complete!")
    return output_path, phis.cpu().numpy().flatten()

def test_image_watermarking():
    """Test function for image watermarking"""
    model_path = "/mnt/wouaf_outputs/exp_1"
    
    # Example usage with different scenarios
    
    # Scenario 1: Random fingerprint
    print("=" * 50)
    print("🎲 Test 1: Random fingerprint")
    print("=" * 50)
    
    # You need to provide an input image path
    input_image_path = "./test_images/input.jpg"  # Change this to your image path
    
    if os.path.exists(input_image_path):
        output_path, fingerprint = embed_watermark_in_image(
            image_path=input_image_path,
            model_path=model_path,
            strength=0.3  # Light modification
        )
    else:
        print(f"⚠️ Input image not found at {input_image_path}")
        print("📝 Please provide a valid image path to test watermarking")
    
    # Scenario 2: Custom fingerprint
    print("\n" + "=" * 50)
    print("🎯 Test 2: Custom fingerprint")
    print("=" * 50)
    
    # Define a custom 32-bit fingerprint
    custom_bits = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1,
                   0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    
    if os.path.exists(input_image_path):
        output_path, fingerprint = embed_watermark_in_image(
            image_path=input_image_path,
            model_path=model_path,
            custom_fingerprint=custom_bits,
            output_path="./test_results/custom_watermarked.png",
            strength=0.5  # Medium modification
        )

if __name__ == "__main__":
    test_image_watermarking()