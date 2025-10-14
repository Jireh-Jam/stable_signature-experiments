# Advanced Watermark Attacks Module

This module provides a comprehensive suite of attack methods to test the robustness of image watermarks. It includes traditional image processing attacks, frequency domain attacks, and state-of-the-art AI-based attacks using diffusion models.

## Overview

The advanced attacks module is designed to evaluate watermark robustness by applying various transformations and modifications that might occur during real-world image sharing, compression, or malicious removal attempts.

## Architecture

```
advanced_attacks/
├── __init__.py           # Module initialization
├── attacks.py            # Main WatermarkAttacker class
├── attack_types.py       # Attack type definitions and configurations
├── frequency_attacks.py  # Frequency domain attack implementations
├── diffusion_attacks.py  # Diffusion model-based attacks
└── run.py               # Command-line interface
```

### Data Flow

```
Input Image → Attack Selection → Attack Application → Output Image
     ↓              ↓                    ↓                ↓
   Metrics ← Quality Assessment ← Feature Analysis ← Comparison
```

## Key Modules & Responsibilities

- **attacks.py**: Main attack orchestration and basic attack implementations
  - Coordinates different attack types
  - Handles image format conversions
  - Provides batch evaluation capabilities

- **attack_types.py**: Attack configurations and type definitions
  - Defines available attack types as enums
  - Provides configuration dataclasses
  - Includes pre-configured attack suites

- **frequency_attacks.py**: Frequency domain manipulations
  - High-frequency filtering attacks
  - Adaptive frequency attacks
  - Bandpass filtering

- **diffusion_attacks.py**: AI-based regeneration attacks
  - Stable Diffusion inpainting
  - Image-to-image regeneration
  - ReSD pipeline integration

## Supported Attacks

| Attack Type | Description | Key Parameters |
|-------------|-------------|----------------|
| **gaussian_blur** | Applies Gaussian blur filter | kernel_size, sigma |
| **gaussian_noise** | Adds random Gaussian noise | std (standard deviation) |
| **jpeg_compression** | JPEG compression artifacts | quality (1-100) |
| **brightness** | Brightness adjustment | factor (>1 brighter, <1 darker) |
| **contrast** | Contrast adjustment | factor (>1 higher, <1 lower) |
| **rotation** | Image rotation | degrees |
| **scale** | Downscale and upscale | factor (0-1) |
| **crop** | Center crop and resize | ratio (0-1) |
| **high_frequency** | Frequency domain filtering | threshold_percentile, filter_strength |
| **diffusion_inpainting** | Mask and regenerate regions | mask_ratio, prompt |
| **diffusion_regeneration** | Full image regeneration | strength, prompt |

## Quickstart

### Installation

```bash
# Install basic dependencies
pip install -r requirements.txt

# For diffusion attacks (optional)
pip install diffusers transformers accelerate

# For advanced denoising (optional)
pip install bm3d
```

### Minimal Example

```bash
# Apply a single attack
python advanced_attacks/run.py watermarked.png --attack gaussian_blur --param 2.0 --output results/

# Run all standard attacks
python advanced_attacks/run.py watermarked.png --original original.png --all --save-metrics
```

## CLI Usage

```bash
python advanced_attacks/run.py [OPTIONS] watermarked_image

Required arguments:
  watermarked_image     Path to the watermarked image

Optional arguments:
  --original PATH       Path to original image (for metrics)
  --output DIR         Output directory (default: ./attack_results)
  --attack TYPE        Specific attack to apply
  --all                Run all standard attacks
  --param VALUE        Primary attack parameter
  --param2 VALUE       Secondary parameter if needed
  --prompt TEXT        Prompt for diffusion attacks
  --device {cuda,cpu}  Device to use
  --save-metrics       Save metrics to CSV
  --verbose            Enable verbose logging
```

### Examples

```bash
# JPEG compression attack
python advanced_attacks/run.py watermarked.png --attack jpeg --param 50

# High frequency filtering
python advanced_attacks/run.py watermarked.png --attack high_frequency --param 95 --param2 0.8

# Diffusion inpainting with custom prompt
python advanced_attacks/run.py watermarked.png --attack diffusion_inpainting \
    --param 0.3 --prompt "A beautiful landscape photograph"

# Batch evaluation with metrics
python advanced_attacks/run.py watermarked.png --original original.png \
    --all --save-metrics --output evaluation_results/
```

## Python API

### Basic Usage

```python
from advanced_attacks import WatermarkAttacker, AttackConfig

# Initialize attacker
attacker = WatermarkAttacker(device='cuda')

# Load images
import cv2
watermarked = cv2.imread('watermarked.png')

# Create attack configuration
config = AttackConfig.gaussian_blur(kernel_size=5, sigma=2.0)

# Apply attack
attacked = attacker.attack(watermarked, config)

# Save result
cv2.imwrite('attacked.png', attacked)
```

### Batch Evaluation

```python
from advanced_attacks import WatermarkAttacker, get_standard_attack_suite

attacker = WatermarkAttacker()
attacks = get_standard_attack_suite()

# Evaluate multiple attacks
image_pairs = [(original1, watermarked1), (original2, watermarked2)]
results = attacker.batch_evaluate(
    image_pairs,
    attacks,
    output_dir='batch_results/'
)

# Results contain comprehensive metrics for each attack
for result in results:
    print(f"{result['attack_name']}: PSNR={result['psnr_attack']:.2f}")
```

### Custom Attack Pipeline

```python
from advanced_attacks.attacks import WatermarkAttacker
from advanced_attacks.attack_types import AttackConfig, AttackType

# Custom attack sequence
attack_sequence = [
    AttackConfig.gaussian_noise(std=0.02),
    AttackConfig.jpeg_compression(quality=85),
    AttackConfig.high_frequency(threshold=90, strength=0.5)
]

attacker = WatermarkAttacker()
result = watermarked.copy()

for config in attack_sequence:
    result = attacker.attack(result, config)
```

## Configuration

### Attack Suites

The module provides pre-configured attack suites:

```python
from advanced_attacks.attack_types import get_standard_attack_suite, get_aggressive_attack_suite

# Standard suite - typical real-world scenarios
standard = get_standard_attack_suite()

# Aggressive suite - stress testing
aggressive = get_aggressive_attack_suite()
```

### Custom Configuration

```python
# Create custom attack configuration
from advanced_attacks.attack_types import AttackConfig, AttackType

config = AttackConfig(
    attack_type=AttackType.HIGH_FREQUENCY,
    params={
        'threshold_percentile': 97.5,
        'filter_strength': 0.9
    },
    description="Aggressive high-frequency filtering"
)
```

## Data I/O

### Expected Input Format
- Images: PNG, JPEG, or other standard formats
- Color space: BGR (OpenCV format) or RGB (PIL format)
- Size: Any size (will be resized for certain attacks)

### Output Format
- Attacked images: Same format and size as input
- Metrics CSV: Comprehensive metrics including PSNR, SSIM, texture similarities
- Logs: Detailed execution logs in output directory

## Extending the Module

### Adding a New Attack

1. Add attack type to `AttackType` enum in `attack_types.py`:
```python
class AttackType(Enum):
    # ... existing types ...
    MY_NEW_ATTACK = "my_new_attack"
```

2. Implement attack method in appropriate module:
```python
def _my_new_attack(self, img: np.ndarray, param1: float) -> np.ndarray:
    """Implement your attack logic."""
    # Attack implementation
    return modified_img
```

3. Add to attack router in `attacks.py`:
```python
elif attack_type == AttackType.MY_NEW_ATTACK:
    result = self._my_new_attack(image, **params)
```

4. Create configuration helper in `attack_types.py`:
```python
@classmethod
def my_new_attack(cls, param1: float = 1.0) -> 'AttackConfig':
    return cls(
        attack_type=AttackType.MY_NEW_ATTACK,
        params={'param1': param1},
        description=f"My new attack with param1={param1}"
    )
```

## Performance & Reproducibility

### Performance Tips
- Use GPU for diffusion attacks: `--device cuda`
- Batch process images when possible
- Pre-load models for multiple attacks
- Use lower resolution for initial testing

### Reproducibility
- Set random seeds for stochastic attacks
- Save attack configurations with results
- Use version control for attack parameters
- Document model versions used

## Troubleshooting

### Common Issues

**Diffusion models not loading:**
```bash
# Install required packages
pip install diffusers transformers accelerate torch

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Memory errors with diffusion attacks:**
- Reduce image size before processing
- Use CPU instead of GPU for smaller models
- Clear GPU cache between attacks

**Import errors:**
```bash
# Ensure you're in the correct directory
cd /path/to/repo
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Diffusion models not available" | Missing dependencies | Install diffusers package |
| "Failed to load image" | Invalid path or format | Check file path and format |
| "CUDA out of memory" | GPU memory exceeded | Use smaller images or CPU |
| "Attack type not found" | Invalid attack name | Check available types with --help |

## Versioning & Changelog

### Version 0.1.0
- Initial release with comprehensive attack suite
- Refactored from multiple legacy implementations
- Added unified CLI and API
- Improved error handling and logging
- Added batch evaluation capabilities
- Integrated frequency and diffusion attacks
- Added comprehensive metrics calculation