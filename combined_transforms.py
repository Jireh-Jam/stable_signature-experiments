import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import functional as F
from augly.image import functional as aug_functional

# filepath: /home/azureuser/cloudfiles/code/Users/David.Fletcher/ost-embedding-research/Temp/PotentialTransforms/combined_transforms.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
image_root_dir = '/home/azureuser/cloudfiles/code/Users/David.Fletcher/pass/'
output_dir = os.path.join(os.path.dirname(image_root_dir), 'OutputTransformations')
os.makedirs(output_dir, exist_ok=True)

# Normalization constants
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnormalize_img = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

# Utility functions
def load_image(image_path):
    """Load an image from a file path."""
    return Image.open(image_path).convert('RGB')

def save_image(image, filename):
    """Save an image to the output directory."""
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)

def tensor_to_image(tensor):
    """Convert a tensor to a PIL image."""
    transform = transforms.ToPILImage()
    return transform(tensor)

def image_to_tensor(image):
    """Convert a PIL image to a tensor."""
    transform = transforms.ToTensor()
    return transform(image)

# Transformations
def resize_image(image_path, output_path, size=(224, 224)):
    """Resize an image to the specified size and save to output path."""
    image = load_image(image_path)
    transform = transforms.Resize(size)
    transformed_image = transform(image)
    transformed_image.save(output_path)

def random_horizontal_flip(image_path, output_path, p=1.0):
    """Apply random horizontal flip to an image and save to output path."""
    image = load_image(image_path)
    transform = transforms.RandomHorizontalFlip(p=p)
    transformed_image = transform(image)
    transformed_image.save(output_path)

def fixed_rotation(image_path, output_path, degrees=15):
    """Apply fixed rotation to an image and save to output path."""
    image = load_image(image_path)
    transformed_image = transforms.functional.rotate(image, angle=degrees)
    transformed_image.save(output_path)

def color_jitter(image_path, output_path, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    """Apply color jitter to an image and save to output path."""
    image = load_image(image_path)
    transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    transformed_image = transform(image)
    transformed_image.save(output_path)

def normalize_image(image_path, output_path):
    """Normalize an image and save to output path.""" #Code different from image_transforms.py
    image = load_image(image_path)
    tensor = image_to_tensor(image)
    normalized_tensor = normalize_img(tensor)
    normalized_image = tensor_to_image(normalized_tensor)
    normalized_image.save(output_path)

def gaussian_blur(image_path, output_path, kernel_size=51):
    """Apply Gaussian blur to an image and save to output path."""
    image = load_image(image_path)
    transform = transforms.GaussianBlur(kernel_size)
    transformed_image = transform(image)
    transformed_image.save(output_path)

def centre_crop(image_path, output_path, size=(224, 224)):
    """Apply centre crop to an image and save to output path."""
    image = load_image(image_path)
    transform = transforms.CenterCrop(size)
    transformed_image = transform(image)
    transformed_image.save(output_path)

def random_perspective(image_path, output_path, distortion_scale=0.5, p=1.0):
    """Apply random perspective transformation to an image and save to output path."""
    image = load_image(image_path)
    transform = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p)
    transformed_image = transform(image)
    transformed_image.save(output_path)

def random_erasing(image_path, output_path, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    """Apply random erasing to an image and save to output path."""
    image = load_image(image_path)
    tensor = image_to_tensor(image)
    transform = transforms.RandomErasing(p=p, scale=scale, ratio=ratio)
    erased_tensor = transform(tensor)
    erased_image = tensor_to_image(erased_tensor)
    erased_image.save(output_path)

def grayscale(image_path, output_path):
    """Convert an image to grayscale and save to output path."""
    image = load_image(image_path)
    transform = transforms.Grayscale(num_output_channels=3)
    transformed_image = transform(image)
    transformed_image.save(output_path)

def overlay_text(image_path, output_path, text='Sample Text', position=(50, 50), color=(255, 255, 255), font_path=None, font_size=20):
    """Overlay text on an image and save to output path."""
    image = load_image(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    image.save(output_path)

def jpeg_compress(image_path, output_path, quality=85):
    """Apply JPEG compression to an image and save to output path."""
    image = load_image(image_path)
    image.save(output_path, 'JPEG', quality=quality)


def adjust_brightness(image_path, output_path, brightness_factor):
    """Adjust brightness of an image and save to output path."""
    image = load_image(image_path)
    tensor = image_to_tensor(image)
    adjusted_tensor = normalize_img(F.adjust_brightness(unnormalize_img(tensor), brightness_factor))
    adjusted_image = tensor_to_image(adjusted_tensor)
    adjusted_image.save(output_path)

def adjust_contrast(image_path, output_path, contrast_factor):
    """Adjust contrast of an image and save to output path."""
    image = load_image(image_path)
    tensor = image_to_tensor(image)
    adjusted_tensor = normalize_img(F.adjust_contrast(unnormalize_img(tensor), contrast_factor))
    adjusted_image = tensor_to_image(adjusted_tensor)
    adjusted_image.save(output_path)

def adjust_saturation(image_path, output_path, saturation_factor):
    """Adjust saturation of an image and save to output path."""
    image = load_image(image_path)
    tensor = image_to_tensor(image)
    adjusted_tensor = normalize_img(F.adjust_saturation(unnormalize_img(tensor), saturation_factor))
    adjusted_image = tensor_to_image(adjusted_tensor)
    adjusted_image.save(output_path)

def adjust_hue(image_path, output_path, hue_factor):
    """Adjust hue of an image and save to output path."""
    image = load_image(image_path)
    tensor = image_to_tensor(image)
    adjusted_tensor = normalize_img(F.adjust_hue(unnormalize_img(tensor), hue_factor))
    adjusted_image = tensor_to_image(adjusted_tensor)
    adjusted_image.save(output_path)

def adjust_gamma(image_path, output_path, gamma, gain=1):
    """Adjust gamma of an image and save to output path."""
    image = load_image(image_path)
    tensor = image_to_tensor(image)
    adjusted_tensor = normalize_img(F.adjust_gamma(unnormalize_img(tensor), gamma, gain))
    adjusted_image = tensor_to_image(adjusted_tensor)
    adjusted_image.save(output_path)

def adjust_sharpness(image_path, output_path, sharpness_factor):
    """Adjust sharpness of an image and save to output path."""
    image = load_image(image_path)
    tensor = image_to_tensor(image)
    adjusted_tensor = normalize_img(F.adjust_sharpness(unnormalize_img(tensor), sharpness_factor))
    adjusted_image = tensor_to_image(adjusted_tensor)
    adjusted_image.save(output_path)

def bitmask_image(image_path, output_path, bits = 3):
    """Apply bitmask to an image and save to output path."""
    image = load_image(image_path)
    pixels = image.load()
    mask = 0xFF << bits

    for i in range(image.size[0]):
        for j in range(image.size[1]):
            r, g, b = pixels[i, j]
            r = r & mask
            g = g & mask
            b = b & mask
            pixels[i, j] = (r, g, b)

    image.save(output_path)

# Example usage
if __name__ == "__main__":
    image_path = '/home/azureuser/cloudfiles/code/Users/David.Fletcher/pass/0/21657882a4c879e3d08bf7eb59974515.jpg'
    output_dir = '/home/azureuser/cloudfiles/code/Users/David.Fletcher/pass/OutputTransformations'

    # Apply transformations and save directly
    resize_image(image_path, os.path.join(output_dir, 'resized_image.jpg'))
    random_horizontal_flip(image_path, os.path.join(output_dir, 'flipped_image.jpg'))
    fixed_rotation(image_path, os.path.join(output_dir, 'rotated_image.jpg'))
    color_jitter(image_path, os.path.join(output_dir, 'jittered_image.jpg'))
    normalize_image(image_path, os.path.join(output_dir, 'normalized_image.jpg'))
    gaussian_blur(image_path, os.path.join(output_dir, 'blurred_image.jpg'))
    centre_crop(image_path, os.path.join(output_dir, 'cropped_image.jpg'))
    random_perspective(image_path, os.path.join(output_dir, 'perspective_image.jpg'))
    random_erasing(image_path, os.path.join(output_dir, 'erased_image.jpg'))
    grayscale(image_path, os.path.join(output_dir, 'grayscale_image.jpg'))
    overlay_text(image_path, os.path.join(output_dir, 'text_overlay_image.jpg'), text="Hello World", position=(10, 10), color=(255, 0, 0))
    jpeg_compress(image_path, os.path.join(output_dir, 'compressed_image.jpg'), quality=50)
    adjust_brightness(image_path, os.path.join(output_dir, 'brightness_adjusted_image.jpg'), brightness_factor=1.5)
    adjust_contrast(image_path, os.path.join(output_dir, 'contrast_adjusted_image.jpg'), contrast_factor=1.5)
    adjust_saturation(image_path, os.path.join(output_dir, 'saturation_adjusted_image.jpg'), saturation_factor=1.5)
    adjust_hue(image_path, os.path.join(output_dir, 'hue_adjusted_image.jpg'), hue_factor=0.1)
    adjust_gamma(image_path, os.path.join(output_dir, 'gamma_adjusted_image.jpg'), gamma=2.0, gain=1.0)
    adjust_sharpness(image_path, os.path.join(output_dir, 'sharpness_adjusted_image.jpg'), sharpness_factor=2.0)
    bitmask_image(image_path, os.path.join(output_dir, 'bitmasked_image.jpg'), bits=3)