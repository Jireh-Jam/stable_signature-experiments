# TrustMark

TrustMark is a robust watermarking system with quality-factor tuning capabilities, designed for reliable watermark detection even after significant image modifications.

## Overview

- **Type**: Robust watermarking with quality tuning
- **Package**: Available via pip
- **Key Feature**: Quality-factor tuning for balancing robustness and image quality

## Setup

### 1. Install TrustMark

```bash
pip install trustmark
```

### 2. Prepare Watermarked Images

Generate watermarked images using TrustMark and place them in the `watermarked_images/` directory.

Example encoding:
```python
from trustmark import TrustMark
from PIL import Image

# Initialise TrustMark
tm = TrustMark(verbose=True, model_type='Q')

# Load and encode image
image = Image.open('original.png')
watermark_bits = [1, 0, 1, 1, 0, 1, 0, 0]  # Your watermark
watermarked_image = tm.encode(image, watermark_bits)

# Save watermarked image
watermarked_image.save('watermarked_images/example.png')
```

## Usage in Pipeline

The pipeline will:
1. Initialise TrustMark with model type 'Q' (quality model)
2. Process images from `watermarked_images/`
3. Use the `decode` method to extract watermarks
4. Test detection robustness across transformations

## Model Types

TrustMark offers different model types:
- **'Q' (Quality)**: Prioritises image quality
- **'R' (Robustness)**: Prioritises watermark robustness

The pipeline uses the 'Q' model by default for a balance of quality and detection.

## Key Features

- **Easy to Use**: Simple Python API
- **No External Models**: Built-in model weights
- **Flexible**: Adjustable quality-robustness trade-off
- **Fast**: Efficient encoding and decoding

## Watermark Detection

TrustMark's `decode` method returns:
- **detected_bits**: Array of detected watermark bits
- **confidence**: Detection confidence score

Example:
```python
detected_bits, confidence = tm.decode(watermarked_image)
print(f"Detected: {detected_bits}")
print(f"Confidence: {confidence}")
```

## Performance Characteristics

- **Embedding Capacity**: Typically 32-256 bits
- **Robustness**: Good resistance to JPEG compression, resizing, and noise
- **Quality Impact**: Minimal visual distortion (PSNR > 40 dB typical)

## Troubleshooting

**Issue**: Low detection confidence
- **Solution**: Check image quality and consider using the 'R' model type

**Issue**: Import errors
- **Solution**: Ensure TrustMark is properly installed: `pip install --upgrade trustmark`

## Additional Resources

For more information:
- TrustMark documentation
- Example notebooks in the TrustMark repository
- Research papers on robust watermarking techniques
