# Watermark Detector Module

This module provides functionality for detecting and extracting watermarks from images using pre-trained HiDDeN (Hidden Deep Neural Networks) models. It supports single image detection, batch processing, and watermark verification.

## Purpose and Scope

The detector module is designed to:
- Extract hidden watermark messages from images
- Verify the presence of expected watermarks
- Evaluate watermark robustness after attacks
- Process large batches of images efficiently
- Provide confidence scores for detections

## Model Architecture

The detector uses the HiDDeN decoder architecture:

```
Input Image (512x512) → Encoder CNN → Feature Extraction → Decoder CNN → Binary Message (48 bits)
                                           ↓
                                    Confidence Score
```

### Supported Models

| Model | Input Type | Output | Default Params |
|-------|------------|--------|----------------|
| HiddenDecoder | RGB Image (512x512) | 48-bit message | depth=8, channels=64 |
| EncoderWithJND | RGB Image + Message | Watermarked Image | JND attenuation |

## Pipeline Overview

### Detection Pipeline

1. **Preprocessing**
   - Load and resize image to 512x512
   - Normalize using ImageNet statistics
   - Convert to tensor

2. **Inference**
   - Forward pass through decoder network
   - Extract logits for each bit
   - Apply threshold (>0) for binary message

3. **Postprocessing**
   - Convert binary message to string
   - Calculate confidence score
   - Optional verification against expected message

## Usage

### CLI Examples

#### Single Image Detection

```bash
# Basic detection
python detector/run.py watermarked_image.png

# With expected message verification
python detector/run.py watermarked_image.png \
    --expected-message "111010110101000001010111010011010100010000100111"

# Custom checkpoint
python detector/run.py watermarked_image.png \
    --checkpoint path/to/model.pth
```

#### Batch Processing

```bash
# Process entire directory
python detector/run.py /path/to/images/ --output results.csv

# With custom batch size
python detector/run.py /path/to/images/ \
    --batch-size 16 \
    --output results.csv
```

### Python API

#### Basic Detection

```python
from detector import WatermarkDetector

# Initialize detector
detector = WatermarkDetector(
    checkpoint_path="ckpts/hidden_replicate.pth",
    device="cuda"
)

# Detect watermark
message = detector.detect("watermarked_image.png")
print(f"Detected message: {message}")

# Get confidence score
message, confidence = detector.detect("watermarked_image.png", return_confidence=True)
print(f"Message: {message}, Confidence: {confidence:.4f}")
```

#### Verification

```python
# Verify expected watermark
expected_msg = "111010110101000001010111010011010100010000100111"
is_verified, accuracy, bit_errors = detector.verify_watermark(
    "watermarked_image.png",
    expected_msg,
    threshold=0.75
)

if is_verified:
    print(f"Watermark verified! Accuracy: {accuracy:.2%}")
else:
    print(f"Verification failed. Bit errors: {bit_errors['error_bits']}")
```

#### Batch Processing

```python
# Process multiple images
images = ["img1.png", "img2.png", "img3.png"]
results = detector.detect_batch(images, batch_size=8)

for result in results:
    print(f"Image {result['image_index']}: {result['message']}")
```

#### Robustness Evaluation

```python
# Evaluate against attacks
attacked_images = {
    "jpeg_compressed": "attacked_jpeg.png",
    "gaussian_blur": "attacked_blur.png",
    "rotation": "attacked_rotation.png"
}

results = detector.evaluate_robustness(
    "original_watermarked.png",
    attacked_images,
    expected_msg
)

for attack, metrics in results.items():
    print(f"{attack}: Accuracy={metrics['accuracy']:.2%}")
```

## Configuration

### Model Parameters

Create a configuration file `config.json`:

```json
{
    "encoder_depth": 4,
    "encoder_channels": 64,
    "decoder_depth": 8,
    "decoder_channels": 64,
    "num_bits": 48,
    "attenuation": "jnd",
    "scale_channels": false,
    "scaling_i": 1.0,
    "scaling_w": 1.5
}
```

Use with:
```bash
python detector/run.py image.png --config config.json
```

### Threshold Settings

- **Verification threshold**: Default 0.75 (75% bit accuracy)
- **Confidence calculation**: Average absolute logit values
- **Binary threshold**: 0 (positive logits → bit 1, negative → bit 0)

## Evaluation

### Running Evaluations

```bash
# Evaluate detection accuracy on a dataset
python detector/run.py /path/to/test/images/ \
    --expected-message "YOUR_MESSAGE" \
    --threshold 0.8 \
    --output evaluation_results.csv
```

### Metrics

The detector provides several metrics:

- **Bit Accuracy**: Percentage of correctly detected bits
- **Bit Error Rate (BER)**: 1 - accuracy
- **Confidence Score**: Average magnitude of decoder outputs
- **Verification Status**: Pass/fail based on threshold

### Interpreting Results

| Confidence | Interpretation |
|------------|----------------|
| > 10.0 | Very strong watermark |
| 5.0 - 10.0 | Strong watermark |
| 2.0 - 5.0 | Moderate watermark |
| < 2.0 | Weak/uncertain watermark |

## Deployment Notes

### Model Loading

```python
# Custom checkpoint loading
from detector.utils import Params
from detector import WatermarkDetector

# Load with custom parameters
params = Params(
    decoder_depth=10,
    decoder_channels=128,
    num_bits=64
)

detector = WatermarkDetector(
    checkpoint_path="custom_model.pth",
    params=params
)
```

### ONNX Export (Optional)

```python
# Export to ONNX for deployment
import torch

# Get model
model = detector.decoder

# Create dummy input
dummy_input = torch.randn(1, 3, 512, 512).to(detector.device)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "watermark_decoder.onnx",
    opset_version=11,
    input_names=['image'],
    output_names=['message_logits']
)
```

### Performance Optimization

- **GPU Batching**: Use batch_size=16 or 32 for GPU
- **CPU Threading**: Set OMP_NUM_THREADS for CPU inference
- **Mixed Precision**: Use fp16 on compatible GPUs
- **Image Preprocessing**: Pre-resize images to 512x512

## Troubleshooting

### Common Issues

**"Checkpoint not found"**
```bash
# Check file path
ls -la ckpts/hidden_replicate.pth

# Download checkpoint if missing
wget https://example.com/checkpoint.pth -O ckpts/hidden_replicate.pth
```

**"CUDA out of memory"**
```bash
# Use CPU instead
python detector/run.py image.png --device cpu

# Or reduce batch size
python detector/run.py images/ --batch-size 4
```

**"Size mismatch in model"**
- Check that checkpoint matches the model parameters
- Verify num_bits, decoder_depth, decoder_channels settings

**Low detection accuracy**
- Ensure image is properly watermarked
- Check image preprocessing (resize, normalization)
- Verify checkpoint is from correct training

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Inspect intermediate values
detector = WatermarkDetector(checkpoint_path, device="cpu")
with torch.no_grad():
    # Get raw logits
    logits = detector.decoder(image_tensor)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
```

## FAQs

**Q: What image formats are supported?**
A: PNG, JPEG, and most standard formats via PIL.

**Q: Can I detect multiple watermarks in one image?**
A: No, the current model detects a single 48-bit watermark per image.

**Q: How robust is the watermark to attacks?**
A: Robustness varies by attack. Use `evaluate_robustness()` to test.

**Q: Can I change the watermark message length?**
A: Yes, but you need to retrain the model with the new bit length.

**Q: Is the detector rotation-invariant?**
A: No, significant rotations will affect detection accuracy.

## Integration Examples

### With Advanced Attacks Module

```python
from detector import WatermarkDetector
from advanced_attacks import WatermarkAttacker, get_standard_attack_suite

# Attack and detect
attacker = WatermarkAttacker()
detector = WatermarkDetector("ckpts/hidden_replicate.pth")

# Apply attacks and verify watermark survives
for name, config in get_standard_attack_suite().items():
    attacked = attacker.attack(watermarked_img, config)
    message, conf = detector.detect(attacked, return_confidence=True)
    print(f"{name}: {message} (conf: {conf:.2f})")
```

### With Watermark Generation

```python
# Full pipeline example
from detector import WatermarkDetector
import generate_watermarked_images as gen

# Generate watermarked image
# ... (generation code)

# Verify watermark
detector = WatermarkDetector("ckpts/hidden_replicate.pth")
detected_msg = detector.detect("output/watermarked/image.png")
assert detected_msg == original_msg, "Watermark verification failed!"
```