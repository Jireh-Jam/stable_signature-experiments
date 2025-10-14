import torch
import os
import pandas as pd
import cv2
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
from torchvision import transforms
import numpy as np
from models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from attenuations import JND

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper functions to convert between boolean message arrays and bit strings
def msg2str(msg):
    return "".join(['1' if el else '0' for el in msg])


def str2msg(s):
    return [True if el == '1' else False for el in s]


# Parameters class
class Params():
    def __init__(self, encoder_depth: int, encoder_channels: int, decoder_depth: int, decoder_channels: int,
                 num_bits: int,
                 attenuation: str, scale_channels: bool, scaling_i: float, scaling_w: float):
        # Encoder and decoder parameters
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        # Attenuation parameters
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w


# Define image transforms (using ImageNet normalization)
NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
default_transform = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE_IMAGENET
])

# Set up parameters
params = Params(
    encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
    attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
)

# Create encoder and decoder models
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

# Move encoder and decoder to device
encoder_with_jnd = encoder_with_jnd.to(device).eval()
decoder = decoder.to(device).eval()


# Function to load a decoder from a checkpoint
def load_decoder(ckpt_path, decoder_depth, num_bits, decoder_channels, device):
    """Loads the HiddenDecoder model with weights from a checkpoint."""
    decoder_model = HiddenDecoder(num_blocks=decoder_depth, num_bits=num_bits, channels=decoder_channels)
    state_dict = torch.load(ckpt_path, map_location=device)['encoder_decoder']
    # Remove any "module." prefixes if present
    encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    decoder_state_dict = {k.replace('decoder.', ''): v
                          for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
    decoder_model.load_state_dict(decoder_state_dict)
    decoder_model = decoder_model.to(device).eval()
    return decoder_model


# Function to detect watermark from an image using the decoder model
def detect_watermark(image_path, ckpt_path, decoder_depth=8, num_bits=48, decoder_channels=64):
    """Detects and returns the watermark bits hidden in the image."""
    # Load watermarked image and resize to expected dimensions.
    img = Image.open(image_path).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    # Show the images during loading using cv2
    cv2.imshow('image', np.array(img))
    cv2.waitKey(1000)
    # Apply the default transform.
    img_tensor = default_transform(img).unsqueeze(0).to(device)
    # Load the decoder model.
    decoder_model = load_decoder(ckpt_path, decoder_depth, num_bits, decoder_channels, device)
    # Decode the watermark.
    ft = decoder_model(img_tensor)
    decoded_msg = ft > 0  # Threshold to obtain binary message
    decoded_str = msg2str(decoded_msg.squeeze(0).cpu().numpy())
    return decoded_str


# Function to process multiple images in a folder and save results to CSV
def process_images_in_folder(image_folder, ckpt_path, output_csv="metrics.csv"):
    """Process multiple images, detect watermark, and store metrics in a CSV file."""
    results = []

    # Iterate over images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            print(f"Processing {image_path}...")
            decoded_str = detect_watermark(image_path, ckpt_path)
            # Store the results (you can add more metrics as needed)
            results.append({"image": image_path, "decoded_message": decoded_str})

    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Metrics saved to {output_csv}")


