import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import csv
from datetime import datetime
from tqdm import tqdm
import random
from models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from attenuations import JND

####################################################################################################
# Author: Jireh Jam
# Date: 2021-06-30
# Description: This script demonstrates how to generate watermarked images using a pre-trained
#              HiddenEncoder and HiddenDecoder model. The script processes images from a specified
#              directory and saves the watermarked images, original images, difference images, and  
#              combined visualizations. The script also calculates the PSNR and bit accuracy of the
#              watermarked images.
####################################################################################################

# Function definitions
def msg2str(msg):
    returgn "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]

class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float):
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

def create_directories(base_dir="output"):
    """Create necessary directories for output"""
    directories = {
        'watermarked': os.path.join(base_dir, 'watermarked'),
        'original': os.path.join(base_dir, 'original'),
        'difference': os.path.join(base_dir, 'difference'),
        'combined': os.path.join(base_dir, 'combined'),
        'metrics': os.path.join(base_dir, 'metrics')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def get_all_image_paths(base_dir):
    """Get all image paths from the nested directory structure"""
    image_paths = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(folder_path, img_file)
                    subfolder = os.path.basename(folder_path)
                    image_paths.append({
                        'full_path': full_path,
                        'subfolder': subfolder,
                        'filename': img_file
                    })
    return image_paths

def process_image(image_info, encoder_with_jnd, decoder, params, directories, device, default_transform, random_msg=False):
    """Process a single image with watermarking"""
    # Extract information
    img_path = image_info['full_path']
    subfolder = image_info['subfolder']
    filename = os.path.splitext(image_info['filename'])[0]
    
    # Load and prepare image
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.BICUBIC)
        img_pt = default_transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image {img_path}: {str(e)}")
        raise
    
    # Create message
    if random_msg:
        msg_ori = torch.randint(0, 2, (1, params.num_bits), device=device).bool()
    else:
        msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0).to(device)
    msg = 2 * msg_ori.type(torch.float) - 1
    
    # Encode
    img_w = encoder_with_jnd(img_pt, msg)
    clip_img = torch.clamp(UNNORMALIZE_IMAGENET(img_w), 0, 1)
    clip_img = torch.round(255 * clip_img)/255 
    clip_img = transforms.ToPILImage()(clip_img.squeeze(0).cpu())
    
    # Calculate difference and PSNR
    diff = np.abs(np.asarray(img).astype(int) - np.asarray(clip_img).astype(int)) / 255 * 10
    psnr = peak_signal_noise_ratio(np.array(img), np.array(clip_img))
    
    # Save images with original folder structure
    output_filename = f"{subfolder}_{filename}"
    img.save(os.path.join(directories['original'], f"{output_filename}_original.png"))
    clip_img.save(os.path.join(directories['watermarked'], f"{output_filename}_watermarked.png"))
    
    # Save difference image
    plt.imsave(os.path.join(directories['difference'], f"{output_filename}_difference.png"), diff)
    
    # Create combined visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(clip_img)
    ax2.set_title('Watermarked')
    ax2.axis('off')
    ax3.imshow(diff)
    ax3.set_title('Difference')
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(directories['combined'], f"{output_filename}_combined.png"))
    plt.close()
    
    # Decode and calculate accuracy
    ft = decoder(default_transform(clip_img).unsqueeze(0).to(device))
    decoded_msg = ft > 0
    msg_ori = msg_ori.to(device)  # Ensure msg_ori is on the same device
    accs = (~torch.logical_xor(decoded_msg, msg_ori))
    bit_accuracy = accs.sum().item() / params.num_bits
    
    # Create metrics
    metrics = {
        'subfolder': subfolder,
        'filename': filename,
        'PSNR': psnr,
        'bit_accuracy': bit_accuracy,
        'original_message': msg2str(msg_ori.squeeze(0).cpu().numpy()),
        'decoded_message': msg2str(decoded_msg.squeeze(0).cpu().numpy()),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save individual metrics
    metrics_path = os.path.join(directories['metrics'], f"{output_filename}_metrics.csv")
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    
    return metrics

if __name__ == "__main__":
    # Constants and device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])

    # Initialize parameters
    params = Params(
        encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
        attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
    )

    # Initialize models
    decoder = HiddenDecoder(
        num_blocks=params.decoder_depth, 
        num_bits=params.num_bits, 
        channels=params.decoder_channels
    )
    encoder = HiddenEncoder(
        num_blocks=params.encoder_depth, 
        num_bits=params.num_bits, 
        channels=params.encoder_channels
    )
    attenuation = JND(preprocess=UNNORMALIZE_IMAGENET) if params.attenuation == "jnd" else None
    encoder_with_jnd = EncoderWithJND(
        encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
    )

    # Load model weights
    ckpt_path = "ckpts/hidden_replicate.pth"
    state_dict = torch.load(ckpt_path, map_location='cpu')['encoder_decoder']
    encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
    decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    # Move models to device and set to eval mode
    encoder_with_jnd = encoder_with_jnd.to(device).eval()
    decoder = decoder.to(device).eval()

    # Create output directories
    directories = create_directories()

    # Get all image paths
    input_directory = "../../pass"  # Update this to your pass directory path
    all_image_paths = get_all_image_paths(input_directory)
    num_images = 3000  # Set the number of images to process

    # Randomly select images if needed
    if num_images and num_images < len(all_image_paths):
        selected_images = random.sample(all_image_paths, num_images)
    else:
        selected_images = all_image_paths

    # Process images
    all_metrics = []
    for img_info in tqdm(selected_images, desc="Processing images"):
        try:
            metrics = process_image(
                img_info,
                encoder_with_jnd,
                decoder,
                params,
                directories,
                device,
                default_transform,
                random_msg=False
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {img_info['full_path']}: {str(e)}")
            continue

    # Save summary metrics
    if all_metrics:
        summary_path = os.path.join(directories['metrics'], 'summary_metrics.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)

    print(f"Successfully processed {len(all_metrics)} images")