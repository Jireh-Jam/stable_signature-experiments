# Generate Watermarked Images Script Documentation

## Overview

The `generate_watermarked_images.py` script is a comprehensive tool for embedding invisible watermarks into images using the HiDDeN (Hidden Deep Neural Networks) approach. It processes images from a nested directory structure, applies watermarks, and generates various outputs including difference maps and performance metrics.

## What It Does

The script performs the following operations:

1. **Loads pre-trained HiDDeN models** (encoder and decoder)
2. **Processes images** from a directory structure
3. **Embeds watermarks** using Just Noticeable Difference (JND) perceptual masking
4. **Generates outputs**:
   - Original images (resized to 512x512)
   - Watermarked images
   - Difference maps showing watermark locations
   - Combined visualizations
   - Performance metrics (PSNR, bit accuracy)

## Inputs

### Required Inputs

1. **Input Directory Structure**:
   ```
   input/
   ├── category1/
   │   ├── image1.jpg
   │   ├── image2.png
   │   └── ...
   ├── category2/
   │   ├── image3.jpg
   │   └── ...
   └── ...
   ```

2. **Model Checkpoint**:
   - Default: `ckpts/hidden_replicate.pth`
   - Contains pre-trained encoder and decoder weights

### Optional Inputs

- **Watermark Message**: 48-bit binary string (default: "111010110101000001010111010011010100010000100111")
- **Number of Images**: Maximum images to process
- **Output Directory**: Where to save results (default: "output")

## Outputs

### Directory Structure

```
output/
├── watermarked/          # Watermarked images
├── original/            # Original images (512x512)
├── difference/          # Difference maps
├── combined/            # Side-by-side comparisons
└── metrics/             # Performance metrics
    ├── summary_metrics.csv
    └── individual metrics files
```

### File Naming Convention

For an input file `category/image.jpg`, outputs are named:
- `category_image_original.png`
- `category_image_watermarked.png`
- `category_image_difference.png`
- `category_image_combined.png`
- `category_image_metrics.csv`

### Metrics Generated

Each processed image produces:
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better, typically 40-50 dB)
- **Bit Accuracy**: Percentage of correctly decoded watermark bits
- **Original/Decoded Messages**: For verification

## Usage

### Command Line Interface

```bash
# Basic usage with defaults
python generate_watermarked_images.py

# Custom input/output directories
python generate_watermarked_images.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output

# Process limited number of images
python generate_watermarked_images.py --num-images 100

# Use random watermark messages
python generate_watermarked_images.py --random-msg
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | str | "input" | Base directory containing images |
| `--output-dir` | str | "output" | Directory for outputs |
| `--num-images` | int | None | Max images to process (None = all) |
| `--random-msg` | flag | False | Use random watermark per image |

### Python API Usage

```python
import torch
from PIL import Image
from generate_watermarked_images import (
    process_image, 
    create_directories,
    Params
)

# Setup
params = Params.default()
directories = create_directories("output")

# Load models
encoder_with_jnd = # ... load encoder
decoder = # ... load decoder

# Process single image
image_info = {
    'full_path': 'path/to/image.jpg',
    'subfolder': 'category',
    'filename': 'image.jpg'
}

metrics = process_image(
    image_info,
    encoder_with_jnd,
    decoder,
    params,
    directories,
    device='cuda',
    default_transform=transform,
    random_msg=False
)
```

## Examples

### Example 1: Basic Watermarking

```bash
# Watermark all images in a directory
python generate_watermarked_images.py \
    --input-dir dataset/test_images \
    --output-dir watermarked_results
```

Output:
```
Processing images: 100%|████████| 50/50 [02:30<00:00, 3.00s/it]
Successfully processed 50 images
Average PSNR: 45.32 dB
Average bit accuracy: 100.0%
```

### Example 2: Testing with Random Messages

```bash
# Test with different messages per image
python generate_watermarked_images.py \
    --input-dir samples \
    --num-images 10 \
    --random-msg
```

### Example 3: Integration with Detection

```python
# Generate and immediately verify
from generate_watermarked_images import process_image
from detector import WatermarkDetector

# Generate watermarked image
metrics = process_image(image_info, encoder, decoder, ...)

# Verify watermark
detector = WatermarkDetector("ckpts/hidden_replicate.pth")
detected_msg = detector.detect(metrics['output_path'])
assert detected_msg == metrics['original_message']
```

## Quality and Performance Notes

### Image Quality

- **PSNR Range**: 40-50 dB (excellent quality)
- **Visual Quality**: Watermarks are imperceptible to human eyes
- **JND Masking**: Adapts watermark strength based on visual perception

### Processing Speed

| Device | Images/Second | Notes |
|--------|---------------|-------|
| CPU | 0.5-1 | Intel i7, single thread |
| GPU (GTX 1080) | 5-10 | Batch size 1 |
| GPU (RTX 3090) | 15-25 | Batch size 1 |

### Memory Requirements

- **GPU**: ~2GB VRAM for 512x512 images
- **CPU**: ~4GB RAM recommended
- **Disk**: ~1MB per output image set

## Common Pitfalls and Solutions

### Pitfall 1: Out of Memory Errors

**Problem**: CUDA out of memory when processing
**Solution**: 
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
python generate_watermarked_images.py

# Or process fewer images at once
python generate_watermarked_images.py --num-images 10
```

### Pitfall 2: Wrong Image Dimensions

**Problem**: Input images not 512x512
**Solution**: Script automatically resizes, but be aware:
- Aspect ratios are preserved via center crop
- Very small images may lose quality
- Very large images are downsampled

### Pitfall 3: Missing Checkpoint

**Problem**: "FileNotFoundError: ckpts/hidden_replicate.pth"
**Solution**:
```bash
# Ensure checkpoint exists
mkdir -p ckpts
# Download or copy checkpoint to ckpts/hidden_replicate.pth
```

### Pitfall 4: Inconsistent Bit Accuracy

**Problem**: Decoded message doesn't match original
**Causes**:
- Image was modified after watermarking
- Wrong normalization parameters
- Model/checkpoint mismatch

**Debug**:
```python
# Check intermediate values
print(f"Encoder output range: {img_w.min():.3f} to {img_w.max():.3f}")
print(f"Decoder logits: {ft}")
print(f"Bit-wise comparison: {torch.logical_xor(decoded_msg, msg_ori)}")
```

## Advanced Configuration

### Custom Watermark Messages

```python
# Use specific message
custom_msg = "101010101010101010101010101010101010101010101010"  # 48 bits
msg_tensor = torch.Tensor(str2msg(custom_msg)).unsqueeze(0).to(device)
```

### Batch Processing

```python
# Process in batches for efficiency
def batch_process(image_list, batch_size=8):
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        # Process batch...
```

### Custom Parameters

```python
# Adjust watermark strength
params = Params(
    encoder_depth=4,
    encoder_channels=64,
    decoder_depth=8,
    decoder_channels=64,
    num_bits=48,
    attenuation="jnd",
    scale_channels=False,
    scaling_i=1.0,      # Original image weight
    scaling_w=2.0       # Watermark weight (higher = stronger)
)
```

## Integration with Other Modules

### With Attack Module

```bash
# Generate watermarked images
python generate_watermarked_images.py --output-dir step1_watermarked

# Attack them
python advanced_attacks/run.py \
    step1_watermarked/watermarked/image.png \
    --all --output step2_attacked

# Detect watermarks in attacked images
python detector/run.py step2_attacked/ --output step3_detection.csv
```

### Pipeline Script

```python
# Full pipeline example
import subprocess

# Step 1: Generate
subprocess.run(["python", "generate_watermarked_images.py"])

# Step 2: Attack
subprocess.run(["python", "advanced_attacks/run.py", 
                "output/watermarked/", "--all"])

# Step 3: Evaluate
subprocess.run(["python", "detector/run.py", 
                "attack_results/", "--output", "evaluation.csv"])
```

## Best Practices

1. **Always verify watermarks** after generation
2. **Use consistent parameters** across encode/decode
3. **Keep original images** for comparison
4. **Monitor PSNR values** (should be >40 dB)
5. **Test on diverse images** (different textures, colors)
6. **Document watermark messages** used
7. **Backup model checkpoints** with version info