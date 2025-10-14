# Watermark Detection & Analysis

Advanced watermark detection system with comprehensive evaluation and analysis capabilities.

## Overview

This package provides robust tools for detecting watermarks in images, evaluating detection accuracy, and analyzing watermark robustness. Built on the HiDDeN architecture with support for JND-based perceptual masking and comprehensive batch processing.

## Purpose & Scope

The detector module is designed to:
- **Detect** embedded watermarks in individual images or batches
- **Evaluate** detection accuracy against known ground truth
- **Analyze** watermark robustness under various attacks
- **Provide** comprehensive metrics and reporting
- **Support** research and development of watermarking systems

## Architecture

```
detector/
├── __init__.py              # Package initialization
├── detector.py              # Main WatermarkDetector class
├── models.py               # Model architectures and loading utilities
├── evaluation.py           # Evaluation and analysis tools
├── run.py                  # CLI interface
└── README.md              # This file
```

### Pipeline Overview

```
Input Image → Preprocessing → Model Inference → Postprocessing → Results
     ↓             ↓              ↓               ↓            ↓
  Load &        Normalize    HiDDeN Decoder   Threshold &   Detection
  Resize        & Transform    Network        Decode Bits   Metrics
```

## Models & Features

| Model | Input Type | Output | Key Features | Metrics |
|-------|------------|--------|--------------|---------|
| **HiDDeN Decoder** | RGB Images (512x512) | 48-bit message | CNN-based, robust to attacks | Bit accuracy, confidence scores |
| **JND Attenuation** | RGB Images | Perceptual mask | Luminance & contrast masking | Perceptual quality |
| **Encoder (optional)** | RGB + Message | Watermarked image | For end-to-end testing | PSNR, SSIM |

### Model Architecture Details

- **Decoder**: 8-layer CNN with adaptive pooling and linear projection
- **Encoder**: 4-layer CNN with message concatenation and JND masking
- **Message Length**: 48 bits (configurable)
- **Input Resolution**: 512×512 (automatically resized)
- **Normalization**: ImageNet statistics

## Usage

### Basic Detection

```python
from detector import WatermarkDetector

# Initialize detector
detector = WatermarkDetector(device='cuda')

# Load model
detector.load_model('ckpts/hidden_replicate.pth')

# Detect watermark
result = detector.detect_watermark(
    image='watermarked_image.png',
    expected_message='111010110101000001010111010011010100010000100111'
)

print(f"Detected: {result.detected_message}")
print(f"Accuracy: {result.bit_accuracy:.2%}")
print(f"Confidence: {result.confidence_scores.mean():.3f}")
```

### Batch Processing

```python
# Process multiple images
image_paths = ['img1.png', 'img2.png', 'img3.png']
expected_messages = ['110101...', '001100...', '111000...']

results = detector.detect_batch(
    image_paths=image_paths,
    expected_messages=expected_messages
)

# Analyze results
analysis = detector.analyze_detection_quality(results)
print(f"Success rate: {analysis['success_rate']:.2%}")
print(f"Average accuracy: {analysis['avg_bit_accuracy']:.2%}")
```

### Directory Processing

```python
# Process entire directory
df = detector.process_directory(
    input_dir='dataset/watermarked/',
    output_path='results/detection_results.csv',
    expected_message='111010110101000001010111010011010100010000100111'
)

print(f"Processed {len(df)} images")
print(f"Average accuracy: {df['bit_accuracy'].mean():.2%}")
```

## CLI Usage

### Single Image Detection

```bash
# Basic detection
python -m detector.run detect \
    --image watermarked.png \
    --model ckpts/hidden_replicate.pth \
    --output results/

# With expected message for accuracy
python -m detector.run detect \
    --image watermarked.png \
    --model ckpts/hidden_replicate.pth \
    --expected-message "111010110101000001010111010011010100010000100111" \
    --output results/
```

### Batch Processing

```bash
# Process directory
python -m detector.run batch \
    --input-dir dataset/watermarked/ \
    --model ckpts/hidden_replicate.pth \
    --output results/batch_results.csv \
    --expected-message "111010110101000001010111010011010100010000100111"

# Process with image list
python -m detector.run batch \
    --image-list images.txt \
    --model ckpts/hidden_replicate.pth \
    --output results/
```

### Evaluation

```bash
# Evaluate against attacks
python -m detector.run evaluate \
    --original-dir dataset/original/ \
    --watermarked-dir dataset/watermarked/ \
    --attacked-dir dataset/attacked/ \
    --model ckpts/hidden_replicate.pth \
    --output evaluation/

# Compare multiple models
python -m detector.run compare \
    --models ckpts/model1.pth ckpts/model2.pth \
    --test-dir dataset/test/ \
    --output comparison/
```

## Python API Examples

### Advanced Detection

```python
# Custom configuration
from common.config import Config, ModelParams

config = Config(
    model=ModelParams(
        decoder_depth=8,
        decoder_channels=64,
        num_bits=48
    )
)

detector = WatermarkDetector(config=config)
detector.load_model('ckpts/custom_model.pth')

# Detect with custom target size
result = detector.detect_watermark(
    image='test.png',
    target_size=(256, 256)  # Smaller for faster processing
)
```

### Message Analysis

```python
# Compare detected messages
detected_messages = [result.detected_message for result in results]
expected_message = "111010110101000001010111010011010100010000100111"

comparison = detector.compare_messages(detected_messages, expected_message)

print(f"Perfect matches: {comparison['perfect_matches']}/{comparison['total_messages']}")
print(f"Average Hamming distance: {comparison['avg_hamming_distance']:.1f}")
print(f"Most common message: {comparison['most_common_message']}")
```

### Model Information

```python
# Get model details
info = detector.get_model_info()
print(f"Model type: {info['model_type']}")
print(f"Parameters: {info['parameters']:,}")
print(f"Device: {info['device']}")
```

## Configuration

### Model Parameters

```python
from common.config import ModelParams

params = ModelParams(
    encoder_depth=4,        # Encoder layers (for watermarking)
    encoder_channels=64,    # Encoder channel width
    decoder_depth=8,        # Decoder layers (for detection)
    decoder_channels=64,    # Decoder channel width
    num_bits=48,           # Message length in bits
    attenuation="jnd",     # JND-based perceptual masking
    scale_channels=False,   # Channel scaling for robustness
    scaling_i=1.0,         # Image scaling factor
    scaling_w=1.5          # Watermark scaling factor
)
```

### Detection Thresholds

```python
# Custom thresholding
result = detector.detect_watermark('image.png')
confidence_scores = result.confidence_scores

# Apply custom threshold
threshold = 0.7
custom_bits = confidence_scores > threshold
custom_message = ''.join(['1' if bit else '0' for bit in custom_bits])

print(f"Default message: {result.detected_message}")
print(f"Custom message:  {custom_message}")
```

### Batch Configuration

```python
# Configure batch processing
detector = WatermarkDetector(
    config=config,
    device='cuda'
)

# Process with custom batch size (currently supports 1)
results = detector.detect_batch(
    image_paths=paths,
    batch_size=1,  # Will be expanded in future versions
    target_size=(512, 512)
)
```

## Evaluation

### Robustness Testing

```python
from detector.evaluation import DetectionEvaluator

evaluator = DetectionEvaluator(detector)

# Test against various attacks
robustness_results = evaluator.evaluate_robustness(
    original_images=['orig1.png', 'orig2.png'],
    watermarked_images=['water1.png', 'water2.png'],
    attacked_images=['attack1.png', 'attack2.png'],
    expected_message='111010110101000001010111010011010100010000100111'
)

print(f"Robustness score: {robustness_results['robustness_score']:.2%}")
print(f"Attack resistance: {robustness_results['attack_resistance']}")
```

### Performance Metrics

```python
# Comprehensive evaluation
metrics = evaluator.compute_detection_metrics(
    results=detection_results,
    ground_truth_message='111010110101000001010111010011010100010000100111'
)

print("Detection Metrics:")
print(f"  Bit Error Rate: {metrics['bit_error_rate']:.3f}")
print(f"  Message Error Rate: {metrics['message_error_rate']:.3f}")
print(f"  Confidence Score: {metrics['avg_confidence']:.3f}")
print(f"  Detection Time: {metrics['avg_detection_time']:.3f}s")
```

### ROC Analysis

```python
# ROC curve analysis
roc_data = evaluator.compute_roc_curve(
    confidence_scores=all_confidence_scores,
    ground_truth_bits=ground_truth_bits
)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot(roc_data['fpr'], roc_data['tpr'])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
plt.show()
```

## Model Management

### Loading Models

```python
from detector.models import ModelManager, load_detection_model

# Using ModelManager
manager = ModelManager(model_params, device='cuda')
decoder = manager.load_decoder('ckpts/model.pth')

# Convenience function
decoder = load_detection_model('ckpts/model.pth')

# Load with custom parameters
from common.config import ModelParams
params = ModelParams(num_bits=64, decoder_depth=10)
decoder = load_detection_model('ckpts/model.pth', params)
```

### Model Information

```python
# Inspect model architecture
print(f"Model: {type(decoder).__name__}")
print(f"Parameters: {sum(p.numel() for p in decoder.parameters()):,}")

# Check model configuration
for name, param in decoder.named_parameters():
    print(f"{name}: {param.shape}")
```

### Saving Models

```python
# Save trained model
manager.save_model(
    model=decoder,
    save_path='ckpts/my_model.pth',
    metadata={
        'training_date': '2024-01-01',
        'dataset': 'custom_dataset',
        'accuracy': 0.95
    }
)
```

## Deployment

### Production Inference

```python
# Optimized for production
detector = WatermarkDetector(device='cuda')
detector.load_model('ckpts/production_model.pth')

# Batch processing with error handling
def process_images_safely(image_paths):
    results = []
    for path in image_paths:
        try:
            result = detector.detect_watermark(path)
            if result.success:
                results.append(result)
            else:
                logger.warning(f"Detection failed for {path}: {result.error_message}")
        except Exception as e:
            logger.error(f"Error processing {path}: {str(e)}")
    return results
```

### Model Optimization

```python
# Convert to TorchScript for deployment
decoder.eval()
traced_model = torch.jit.trace(decoder, example_input)
torch.jit.save(traced_model, 'ckpts/model_traced.pt')

# Load traced model
traced_model = torch.jit.load('ckpts/model_traced.pt')
```

### ONNX Export

```python
# Export to ONNX for cross-platform deployment
import torch.onnx

dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    decoder,
    dummy_input,
    'ckpts/model.onnx',
    input_names=['image'],
    output_names=['message_logits'],
    dynamic_axes={
        'image': {0: 'batch_size'},
        'message_logits': {0: 'batch_size'}
    }
)
```

## Troubleshooting & FAQs

### Common Issues

**Q: Model loading fails with "state_dict mismatch"**
```python
# A: Use strict=False for partial loading
decoder.load_state_dict(state_dict, strict=False)

# Or check for 'module.' prefix from DataParallel
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
```

**Q: Detection accuracy is very low**
```python
# A: Check image preprocessing
# Ensure images are properly normalized
from common.image_utils import DEFAULT_TRANSFORM
img_tensor = DEFAULT_TRANSFORM(image)

# Verify model is in eval mode
decoder.eval()

# Check for correct message format
expected = "111010110101000001010111010011010100010000100111"  # 48 bits
assert len(expected) == 48
```

**Q: CUDA out of memory during batch processing**
```python
# A: Reduce batch size or use CPU
detector = WatermarkDetector(device='cpu')

# Or process images one by one
for image_path in image_paths:
    result = detector.detect_watermark(image_path)
    # Process result immediately
```

**Q: Slow detection performance**
```python
# A: Optimize preprocessing
# Use smaller target size for faster processing
result = detector.detect_watermark(
    image='test.png',
    target_size=(256, 256)  # Instead of (512, 512)
)

# Use mixed precision
with torch.autocast(device_type='cuda'):
    features = decoder(img_tensor)
```

### Error Messages

- **"No model loaded"**: Call `detector.load_model()` before detection
- **"Invalid image dimensions"**: Ensure image can be resized to target size
- **"Checkpoint not found"**: Verify model path exists and is accessible
- **"Message length mismatch"**: Check expected message has correct bit length

### Performance Optimization

**Memory Usage**:
```python
# Clear cache between batches
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
decoder.gradient_checkpointing_enable()
```

**Speed Optimization**:
```python
# Compile model (PyTorch 2.0+)
decoder = torch.compile(decoder)

# Use channels_last memory format
img_tensor = img_tensor.to(memory_format=torch.channels_last)
```

### Debugging Detection

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Analyze confidence scores
result = detector.detect_watermark('test.png')
confidence = result.confidence_scores

print(f"Confidence stats:")
print(f"  Mean: {confidence.mean():.3f}")
print(f"  Std:  {confidence.std():.3f}")
print(f"  Min:  {confidence.min():.3f}")
print(f"  Max:  {confidence.max():.3f}")

# Visualize confidence distribution
import matplotlib.pyplot as plt
plt.hist(confidence, bins=20)
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Bit Confidence Distribution')
plt.show()
```

## Integration Examples

### With Attack Evaluation

```python
from advanced_attacks import WatermarkAttacker
from detector import WatermarkDetector

# Initialize both systems
attacker = WatermarkAttacker()
detector = WatermarkDetector()
detector.load_model('ckpts/model.pth')

# Test attack effectiveness
original_msg = "111010110101000001010111010011010100010000100111"

# Apply attack
attack_result = attacker.apply_attack(
    image='watermarked.png',
    attack_name='high_frequency_filter'
)

# Test detection robustness
detection_result = detector.detect_watermark(
    image=attack_result.attacked_image,
    expected_message=original_msg
)

print(f"Attack PSNR: {attack_result.metrics['psnr']:.2f} dB")
print(f"Detection accuracy: {detection_result.bit_accuracy:.2%}")
```

### With Watermark Generation

```python
# End-to-end pipeline
from detector.models import ModelManager

manager = ModelManager(model_params, device='cuda')

# Load encoder and decoder
encoder = manager.load_encoder('ckpts/model.pth')
decoder = manager.load_decoder('ckpts/model.pth')

# Generate watermarked image
message = torch.tensor([[1, 0, 1, 1, ...]])  # 48 bits
watermarked = encoder(original_image_tensor, message)

# Detect watermark
detected_features = decoder(watermarked)
detected_bits = detected_features > 0
accuracy = (detected_bits == message).float().mean()

print(f"Round-trip accuracy: {accuracy:.2%}")
```

---

For more information, see the [main repository README](../README.md) and [generate_watermarked_images documentation](../docs/README_generate_watermarked_images.md).