# Generate Watermarked Images Script

Comprehensive tool for generating watermarked images using the HiDDeN architecture with JND-based perceptual masking.

## Overview

The `generate_watermarked_images.py` script provides an end-to-end solution for embedding watermarks into images while maintaining high perceptual quality. It supports batch processing, comprehensive evaluation metrics, and flexible configuration options.

## Purpose

This script is designed to:
- **Embed** 48-bit watermarks into images using the HiDDeN neural network
- **Evaluate** watermark quality through PSNR, SSIM, and bit accuracy metrics  
- **Process** images in batch with organized directory structure
- **Generate** comprehensive reports and visualizations
- **Support** research and development of watermarking systems

## Watermark Technique

The script uses the **HiDDeN (Hidden Information in Deep Networks)** technique:

1. **Encoder Network**: 4-layer CNN that learns to embed messages imperceptibly
2. **JND Masking**: Just Noticeable Difference-based perceptual masking for optimal placement
3. **Message Encoding**: 48-bit binary messages embedded as floating-point signals
4. **Decoder Network**: 8-layer CNN for robust message extraction
5. **End-to-End Training**: Joint optimization for imperceptibility and robustness

### Technical Details

- **Architecture**: Encoder-Decoder with skip connections and attention mechanisms
- **Message Length**: 48 bits (configurable)
- **Input Resolution**: 512×512 pixels (automatically resized)
- **Perceptual Masking**: Luminance and contrast-based JND modeling
- **Scaling Factors**: Separate scaling for image (1.0) and watermark (1.5) components

## Inputs & Outputs

### Input Requirements

**Image Formats**: PNG, JPEG, BMP, TIFF (any PIL-supported format)

**Directory Structure**:
```
input/
├── folder1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── folder2/
│   ├── image3.jpg
│   └── ...
└── ...
```

**Model Requirements**:
- Pre-trained HiDDeN checkpoint at `ckpts/hidden_replicate.pth`
- Compatible with model parameters: 4-layer encoder, 8-layer decoder, 48-bit messages

### Output Structure

```
output/
├── original/           # Resized original images
│   ├── folder1_image1_original.png
│   ├── folder1_image2_original.png
│   └── ...
├── watermarked/       # Watermarked images
│   ├── folder1_image1_watermarked.png
│   ├── folder1_image2_watermarked.png
│   └── ...
├── difference/        # Amplified difference visualizations
│   ├── folder1_image1_difference.png
│   └── ...
├── combined/          # Side-by-side comparisons
│   ├── folder1_image1_combined.png
│   └── ...
└── metrics/           # Quantitative evaluation
    ├── folder1_image1_metrics.csv
    ├── summary_metrics.csv
    └── ...
```

### File Naming Convention

- **Original**: `{subfolder}_{filename}_original.png`
- **Watermarked**: `{subfolder}_{filename}_watermarked.png`
- **Difference**: `{subfolder}_{filename}_difference.png`
- **Combined**: `{subfolder}_{filename}_combined.png`
- **Metrics**: `{subfolder}_{filename}_metrics.csv`

## CLI Usage & Flags

### Basic Usage

```bash
python generate_watermarked_images.py [OPTIONS]
```

### Command Line Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input-dir` | `str` | `"input"` | Base directory containing images in subfolders |
| `--output-dir` | `str` | `"output"` | Directory to save all outputs |
| `--num-images` | `int` | `None` | Maximum number of images to process (random selection) |
| `--random-msg` | `flag` | `False` | Use random watermark message per image |

### Default Configuration

If no arguments provided, the script uses:
- **Input Directory**: `./input/`
- **Output Directory**: `./output/`
- **Message**: Fixed 48-bit string `"111010110101000001010111010011010100010000100111"`
- **Model Path**: `ckpts/hidden_replicate.pth`
- **Target Size**: 512×512 pixels

## End-to-End Examples

### Example 1: Basic Watermarking

```bash
# Process all images in input/ directory
python generate_watermarked_images.py

# Expected output:
# Processing images: 100%|██████████| 25/25 [00:45<00:00,  1.82s/it]
# Successfully processed 25 images
```

**Result**: All images watermarked with fixed message, comprehensive metrics generated.

### Example 2: Limited Processing with Random Messages

```bash
# Process only 10 images with random messages
python generate_watermarked_images.py \
    --num-images 10 \
    --random-msg \
    --output-dir results/random_test/
```

**Result**: 10 randomly selected images, each with unique watermark message.

### Example 3: Custom Directory Structure

```bash
# Process custom dataset
python generate_watermarked_images.py \
    --input-dir /path/to/dataset/ \
    --output-dir /path/to/results/ \
    --num-images 100
```

**Result**: Processes up to 100 images from custom dataset location.

### Example 4: Programmatic Usage

```python
import subprocess
import sys

# Run with custom parameters
result = subprocess.run([
    sys.executable, 'generate_watermarked_images.py',
    '--input-dir', 'my_images/',
    '--output-dir', 'watermarked_results/',
    '--num-images', '50'
], capture_output=True, text=True)

if result.returncode == 0:
    print("Watermarking completed successfully")
    print(result.stdout)
else:
    print("Error occurred:")
    print(result.stderr)
```

## Quality & Performance Notes

### Quality Metrics

**PSNR (Peak Signal-to-Noise Ratio)**:
- **Typical Range**: 35-45 dB
- **Interpretation**: Higher values indicate better perceptual quality
- **Threshold**: >40 dB considered high quality

**SSIM (Structural Similarity Index)**:
- **Typical Range**: 0.95-0.99
- **Interpretation**: Values closer to 1.0 indicate better structural preservation
- **Threshold**: >0.98 considered excellent quality

**Bit Accuracy**:
- **Typical Range**: 95-100%
- **Interpretation**: Percentage of correctly decoded message bits
- **Threshold**: >98% considered robust watermarking

### Performance Characteristics

**Speed Benchmarks** (NVIDIA RTX 3080):
- **512×512 images**: ~1.8 seconds per image
- **Batch processing**: Linear scaling with image count
- **Memory usage**: ~2GB GPU memory for single image

**Quality vs. Speed Trade-offs**:
```python
# High quality (slower)
scaling_w = 2.0      # Stronger watermark
target_size = (1024, 1024)  # Higher resolution

# Balanced (recommended)
scaling_w = 1.5      # Default watermark strength
target_size = (512, 512)    # Standard resolution

# Fast processing (lower quality)
scaling_w = 1.0      # Weaker watermark
target_size = (256, 256)    # Lower resolution
```

### Device Optimization

**GPU Processing**:
```python
# Automatic GPU detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Manual GPU selection
device = torch.device('cuda:0')  # Use first GPU
```

**CPU Fallback**:
```python
# Force CPU processing (slower but more compatible)
device = torch.device('cpu')
```

**Memory Management**:
```python
# Clear GPU cache between batches
torch.cuda.empty_cache()

# Monitor memory usage
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Failure Modes & Recovery

### Common Failure Scenarios

**1. Model Loading Failure**
```
Error: FileNotFoundError: Checkpoint not found: ckpts/hidden_replicate.pth
```
**Recovery**:
```bash
# Ensure model checkpoint exists
ls -la ckpts/hidden_replicate.pth

# Download if missing
wget https://example.com/hidden_replicate.pth -O ckpts/hidden_replicate.pth
```

**2. Image Loading Failure**
```
Error loading image /path/to/image.jpg: cannot identify image file
```
**Recovery**:
```python
# Validate images before processing
from PIL import Image
try:
    img = Image.open('problematic_image.jpg')
    img.verify()  # Check if image is valid
except Exception as e:
    print(f"Invalid image: {e}")
    # Skip or convert image
```

**3. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```
**Recovery**:
```python
# Reduce batch size or use CPU
device = torch.device('cpu')

# Or process smaller images
target_size = (256, 256)  # Instead of (512, 512)
```

**4. Dimension Mismatch**
```
RuntimeError: Expected input tensor to have shape [1, 3, 512, 512]
```
**Recovery**:
```python
# Ensure proper resizing
img = img.resize((512, 512), Image.BICUBIC)

# Check image mode
if img.mode != 'RGB':
    img = img.convert('RGB')
```

### Error Handling Best Practices

```python
def robust_image_processing(image_path):
    try:
        # Load and validate image
        img = Image.open(image_path).convert('RGB')
        
        # Check dimensions
        if min(img.size) < 64:
            raise ValueError("Image too small")
        
        # Resize safely
        img = img.resize((512, 512), Image.BICUBIC)
        
        return img
        
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {str(e)}")
        return None

# Use in batch processing
successful_images = []
for image_path in image_paths:
    img = robust_image_processing(image_path)
    if img is not None:
        successful_images.append((image_path, img))
```

### Recovery Strategies

**Partial Processing Recovery**:
```python
# Save progress periodically
processed_count = 0
for image_info in image_list:
    try:
        process_image(image_info)
        processed_count += 1
        
        # Save checkpoint every 10 images
        if processed_count % 10 == 0:
            save_checkpoint(processed_count, remaining_images)
            
    except Exception as e:
        logger.error(f"Error processing {image_info}: {e}")
        continue  # Skip failed image, continue with others
```

**Automatic Retry Logic**:
```python
def process_with_retry(image_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            return process_image(image_path)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise
            else:
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                time.sleep(1)  # Brief pause before retry
```

## Integration Examples

### With Attack Evaluation

```python
# Generate watermarked images then test robustness
import subprocess
from advanced_attacks import WatermarkAttacker

# Step 1: Generate watermarked images
subprocess.run([
    'python', 'generate_watermarked_images.py',
    '--input-dir', 'test_images/',
    '--output-dir', 'watermarked_test/',
    '--num-images', '20'
])

# Step 2: Test attack robustness
attacker = WatermarkAttacker()
for watermarked_path in Path('watermarked_test/watermarked/').glob('*.png'):
    original_path = Path('watermarked_test/original/') / watermarked_path.name.replace('_watermarked', '_original')
    
    result = attacker.apply_attack(
        image=watermarked_path,
        attack_name='high_frequency_filter',
        original_image=original_path
    )
    
    print(f"Attack PSNR: {result.metrics['psnr']:.2f} dB")
```

### With Detection Pipeline

```python
# Generate and immediately test detection
from detector import WatermarkDetector

# Generate watermarked images
# ... (run generate_watermarked_images.py)

# Test detection accuracy
detector = WatermarkDetector()
detector.load_model('ckpts/hidden_replicate.pth')

expected_message = "111010110101000001010111010011010100010000100111"
watermarked_dir = Path('output/watermarked/')

detection_results = []
for img_path in watermarked_dir.glob('*.png'):
    result = detector.detect_watermark(
        image=img_path,
        expected_message=expected_message
    )
    detection_results.append(result)

# Analyze detection quality
avg_accuracy = np.mean([r.bit_accuracy for r in detection_results if r.bit_accuracy])
print(f"Average detection accuracy: {avg_accuracy:.2%}")
```

### Batch Processing Pipeline

```python
import os
from pathlib import Path
import pandas as pd

def watermark_dataset(dataset_path, output_path, batch_size=100):
    """Process large dataset in batches."""
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Get all images
    all_images = list(dataset_path.rglob('*.jpg')) + list(dataset_path.rglob('*.png'))
    
    # Process in batches
    results = []
    for i in range(0, len(all_images), batch_size):
        batch_images = all_images[i:i+batch_size]
        
        # Create temporary input directory for batch
        temp_input = output_path / f'temp_input_batch_{i//batch_size}'
        temp_input.mkdir(parents=True, exist_ok=True)
        
        # Copy batch images
        for j, img_path in enumerate(batch_images):
            temp_img_path = temp_input / f'batch_{j}_{img_path.name}'
            shutil.copy2(img_path, temp_img_path)
        
        # Run watermarking
        batch_output = output_path / f'batch_{i//batch_size}'
        subprocess.run([
            'python', 'generate_watermarked_images.py',
            '--input-dir', str(temp_input),
            '--output-dir', str(batch_output)
        ])
        
        # Collect results
        metrics_file = batch_output / 'metrics' / 'summary_metrics.csv'
        if metrics_file.exists():
            batch_df = pd.read_csv(metrics_file)
            results.append(batch_df)
        
        # Cleanup temporary files
        shutil.rmtree(temp_input)
    
    # Combine all results
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(output_path / 'complete_results.csv', index=False)
        
    return final_df

# Usage
results = watermark_dataset('large_dataset/', 'watermarked_output/')
print(f"Processed {len(results)} images")
print(f"Average PSNR: {results['PSNR'].mean():.2f} dB")
```

---

For more information, see the [main repository README](../README.md) and [transformations pipeline documentation](README_transformations_pipeline.md).