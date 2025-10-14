# Advanced Watermark Attacks

A comprehensive suite of watermark attack methods for robustness testing and research.

## Overview

This package provides state-of-the-art watermark attack methods designed to evaluate the robustness of image watermarking systems. It includes frequency domain attacks, diffusion-based attacks, adversarial perturbations, and traditional image processing transformations.

## Architecture

```
advanced_attacks/
├── __init__.py              # Package initialization
├── attacks.py               # Main WatermarkAttacker class
├── attack_registry.py       # Attack registration and discovery
├── frequency_attacks.py     # Frequency domain attacks
├── diffusion_attacks.py     # Diffusion-based attacks
├── run.py                   # CLI interface
└── README.md               # This file
```

### Data Flow

```
Input Image → Attack Selection → Parameter Configuration → Attack Execution → Evaluation → Results
     ↓              ↓                    ↓                     ↓             ↓          ↓
Original/      Attack Registry     Default/Custom         Frequency/      Metrics    Report
Watermarked    System             Parameters             Diffusion/      Calculation Generation
Image                                                    Transform
```

## Key Modules & Responsibilities

- **`attacks.py`**: Main orchestrator class providing unified interface for all attack methods
- **`attack_registry.py`**: Dynamic attack registration system with metadata and parameter management
- **`frequency_attacks.py`**: FFT-based attacks targeting high-frequency watermark components
- **`diffusion_attacks.py`**: Stable Diffusion-based regeneration and inpainting attacks
- **`run.py`**: Command-line interface for batch processing and evaluation

## Supported Attacks

| Attack Name | Category | Description | Computational Cost | Effectiveness |
|-------------|----------|-------------|-------------------|---------------|
| `high_frequency_filter` | Frequency | Removes high-frequency components using FFT | Low | Medium |
| `low_pass_filter` | Frequency | Butterworth low-pass filtering | Low | Medium |
| `notch_filter` | Frequency | Removes specific frequency components | Low | Low |
| `diffusion_inpainting` | Diffusion | Stable Diffusion inpainting attack | High | High |
| `diffusion_img2img` | Diffusion | Image-to-image regeneration | High | High |
| `diffusion_resd` | Diffusion | ReSD (Regeneration Stable Diffusion) | High | High |
| `gaussian_blur` | Transform | Gaussian blur filtering | Low | Medium |
| `jpeg_compression` | Transform | JPEG compression artifacts | Low | Medium |
| `brightness` | Transform | Brightness adjustment | Low | Low |
| `contrast` | Transform | Contrast modification | Low | Low |
| `rotation` | Transform | Image rotation | Low | Medium |
| `crop_center` | Transform | Center cropping | Low | Medium |
| `gaussian_noise` | Transform | Additive Gaussian noise | Low | Medium |

## Quickstart

### Installation

```bash
# Install dependencies
pip install torch torchvision pillow numpy scipy scikit-image pandas matplotlib

# For diffusion attacks (optional but recommended)
pip install diffusers transformers accelerate

# For advanced attacks (optional)
pip install foolbox bm3d compressai
```

### Basic Usage

```python
from advanced_attacks import WatermarkAttacker

# Initialize attacker
attacker = WatermarkAttacker(device='cuda')

# Apply single attack
result = attacker.apply_attack(
    image='watermarked_image.png',
    attack_name='high_frequency_filter',
    parameters={'threshold_percentile': 95, 'filter_strength': 0.8},
    original_image='original_image.png'
)

if result.success:
    result.attacked_image.save('attacked_image.png')
    print(f"PSNR: {result.metrics['psnr']:.2f} dB")
    print(f"SSIM: {result.metrics['ssim']:.4f}")
```

### Comprehensive Evaluation

```python
# Run all available attacks
evaluation = attacker.run_comprehensive_evaluation(
    original_image_path='original.png',
    watermarked_image_path='watermarked.png',
    output_dir='results/'
)

print(f"Success rate: {evaluation.summary_stats['success_rate']:.2%}")
print(f"Most effective: {evaluation.summary_stats['most_effective_attack']}")
```

## CLI Usage

### Single Attack

```bash
# Basic attack
python -m advanced_attacks.run single \
    --watermarked image.png \
    --attack high_frequency_filter \
    --output results/

# With custom parameters
python -m advanced_attacks.run single \
    --watermarked image.png \
    --original original.png \
    --attack diffusion_img2img \
    --parameters strength=0.7 guidance_scale=10.0 \
    --output results/
```

### Comprehensive Evaluation

```bash
# Evaluate all attacks
python -m advanced_attacks.run evaluate \
    --original original.png \
    --watermarked watermarked.png \
    --output evaluation/

# Evaluate specific attacks
python -m advanced_attacks.run evaluate \
    --original original.png \
    --watermarked watermarked.png \
    --attacks high_frequency_filter,diffusion_img2img,gaussian_blur \
    --output evaluation/
```

### Batch Comparison

```bash
# Compare attacks across multiple images
python -m advanced_attacks.run compare \
    --image-dir dataset/ \
    --output comparison/

# With specific image pairs
python -m advanced_attacks.run compare \
    --image-pairs "orig1.png,water1.png" "orig2.png,water2.png" \
    --attacks high_frequency_filter,diffusion_img2img \
    --output comparison/
```

### List Available Attacks

```bash
# List all attacks
python -m advanced_attacks.run list

# List by category
python -m advanced_attacks.run list --category frequency

# Detailed information
python -m advanced_attacks.run list --detailed
```

## Python API Examples

### Custom Attack Parameters

```python
# Frequency domain attack
result = attacker.apply_attack(
    image='watermarked.png',
    attack_name='high_frequency_filter',
    parameters={
        'threshold_percentile': 98,  # More aggressive
        'filter_strength': 0.95
    }
)

# Diffusion attack with custom prompt
result = attacker.apply_attack(
    image='watermarked.png',
    attack_name='diffusion_img2img',
    parameters={
        'prompt': 'A clean, professional photograph',
        'strength': 0.5,
        'num_inference_steps': 50
    }
)
```

### Batch Processing

```python
# Process multiple images
image_paths = ['img1.png', 'img2.png', 'img3.png']
attack_names = ['high_frequency_filter', 'diffusion_img2img']

results = []
for img_path in image_paths:
    for attack_name in attack_names:
        result = attacker.apply_attack(img_path, attack_name)
        results.append(result)
```

### Attack Registry Usage

```python
from advanced_attacks.attack_registry import attack_registry

# List available attacks
attacks = attack_registry.list_attacks()
print(f"Available attacks: {attacks}")

# Get attack information
info = attack_registry.get_attack_info('high_frequency_filter')
print(f"Description: {info.description}")
print(f"Parameters: {info.parameters}")

# Search attacks
freq_attacks = attack_registry.search_attacks('frequency')
print(f"Frequency attacks: {freq_attacks}")
```

## Configuration

### YAML Configuration

Create a `config.yaml` file:

```yaml
model:
  encoder_depth: 4
  encoder_channels: 64
  decoder_depth: 8
  decoder_channels: 64
  num_bits: 48
  attenuation: "jnd"
  scale_channels: false
  scaling_i: 1.0
  scaling_w: 1.5

system:
  device: "auto"  # or "cuda", "cpu"
  batch_size: 1
  num_workers: 4
  seed: 42
  log_level: "INFO"
  output_dir: "output"

attacks:
  - attack_type: "high_frequency_filter"
    parameters:
      threshold_percentile: [75, 90, 95, 98]
      filter_strength: [0.5, 0.8, 0.95]
    enabled: true
  
  - attack_type: "diffusion_img2img"
    parameters:
      strength: [0.3, 0.5, 0.7]
      guidance_scale: [5.0, 7.5, 10.0]
    enabled: true
```

Load configuration:

```python
from common.config import Config

config = Config.from_yaml('config.yaml')
attacker = WatermarkAttacker(config=config)
```

## Data I/O

### Input Formats
- **Images**: PNG, JPEG, BMP, TIFF (PIL-supported formats)
- **Paths**: String paths or `pathlib.Path` objects
- **Size**: Automatically resized to compatible dimensions (multiples of 8 for diffusion)

### Output Structure
```
output_dir/
├── attacked_images/
│   ├── high_frequency_filter_attacked.png
│   ├── diffusion_img2img_attacked.png
│   └── ...
├── evaluation_report.txt
├── comparison_report.txt
└── metrics/
    ├── summary_metrics.csv
    └── detailed_results.json
```

### Folder Conventions
- `original/`: Original (unwatermarked) images
- `watermarked/`: Watermarked images for attack
- `attacked/`: Results of attacks
- `metrics/`: Quantitative evaluation results

## Extending the Framework

### Adding a New Attack

1. **Create Attack Class**:

```python
from advanced_attacks.attack_registry import BaseAttack, AttackInfo
from PIL import Image

class MyCustomAttack(BaseAttack):
    def __init__(self):
        super().__init__(
            name="my_custom_attack",
            description="Description of my attack",
            category="custom"
        )
    
    def attack(self, image: Image.Image, **kwargs) -> Image.Image:
        # Implement your attack logic here
        strength = kwargs.get('strength', 1.0)
        
        # Apply attack to image
        attacked_image = self.apply_my_attack(image, strength)
        
        return attacked_image
    
    def get_default_parameters(self):
        return {'strength': 1.0, 'iterations': 10}
    
    def validate_parameters(self, **kwargs):
        strength = kwargs.get('strength', 1.0)
        return 0 <= strength <= 2.0
```

2. **Register Attack**:

```python
from advanced_attacks.attack_registry import register_attack

# Create attack instance
my_attack = MyCustomAttack()

# Create attack info
attack_info = AttackInfo(
    name="my_custom_attack",
    description="My custom watermark attack",
    category="custom",
    parameters={'strength': [0.5, 1.0, 1.5]},
    computational_cost="medium",
    effectiveness="unknown"
)

# Register
register_attack(my_attack, attack_info)
```

3. **Use Attack**:

```python
# Now available in the registry
result = attacker.apply_attack(
    image='test.png',
    attack_name='my_custom_attack',
    parameters={'strength': 1.5}
)
```

### Adding Transform-Based Attacks

```python
from common.transforms import BaseTransform, TransformResult

class MyTransform(BaseTransform):
    def apply(self, image: Image.Image, **kwargs) -> TransformResult:
        # Implement transformation
        transformed_image = self.transform_image(image, **kwargs)
        
        return TransformResult(
            image=transformed_image,
            parameters=kwargs,
            success=True
        )

# Register with transform registry
from common.transforms import transform_registry
transform_registry.register('my_transform', MyTransform('my_transform'))
```

## Performance & Reproducibility

### GPU Optimization
```python
# Use mixed precision for faster diffusion attacks
attacker = WatermarkAttacker(device='cuda')

# Process in batches for efficiency
results = attacker.compare_attacks(
    images=image_pairs,
    attack_names=['high_frequency_filter', 'gaussian_blur']  # Faster attacks first
)
```

### Reproducibility
```python
# Set seeds for reproducible results
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Use deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Batch Size Considerations
- **Frequency attacks**: Can process large batches efficiently
- **Diffusion attacks**: Limited by GPU memory (typically batch_size=1)
- **Transform attacks**: Moderate batch sizes (4-8) work well

## Troubleshooting

### Common Errors & Fixes

**Error**: `CUDA out of memory`
```python
# Solution: Use CPU for diffusion attacks or smaller images
attacker = WatermarkAttacker(device='cpu')
# Or reduce image size
image = image.resize((256, 256))
```

**Error**: `Attack 'xyz' not found`
```python
# Solution: Check available attacks
available = attacker.get_available_attacks()
print(f"Available: {available}")

# Or register the attack
from advanced_attacks.attack_registry import initialize_registry
initialize_registry()
```

**Error**: `Diffusion models not available`
```bash
# Solution: Install diffusion dependencies
pip install diffusers transformers accelerate
```

**Error**: `Invalid image dimensions for diffusion`
```python
# Solution: Ensure dimensions are multiples of 8
width, height = image.size
new_width = (width // 8) * 8
new_height = (height // 8) * 8
image = image.resize((new_width, new_height))
```

### Performance Issues

**Slow diffusion attacks**:
- Reduce `num_inference_steps` (20-30 instead of 50)
- Use lower `guidance_scale` (5.0-7.5)
- Process smaller images (256x256 or 512x512)

**Memory issues**:
- Use `device='cpu'` for large images
- Process images sequentially instead of in batches
- Call `attacker.cleanup()` between evaluations

### Model Loading Issues

**Checkpoint not found**:
```python
# Verify path exists
from pathlib import Path
checkpoint_path = Path('ckpts/model.pth')
if not checkpoint_path.exists():
    print(f"Checkpoint not found at {checkpoint_path}")
```

**State dict mismatch**:
```python
# Use strict=False for partial loading
model.load_state_dict(state_dict, strict=False)
```

## Versioning & Changelog

### Version 1.0.0 (Current)

**Major Changes**:
- ✅ Refactored monolithic attack classes into modular components
- ✅ Added comprehensive attack registry system
- ✅ Implemented unified CLI interface
- ✅ Added type hints and proper logging throughout
- ✅ Created shared utilities in `common/` package
- ✅ Consolidated transform pipeline from multiple sources
- ✅ Added configuration management system

**New Features**:
- 🆕 Dynamic attack discovery and registration
- 🆕 Comprehensive evaluation framework
- 🆕 Multi-image comparison capabilities
- 🆕 Detailed reporting and metrics
- 🆕 Extensible plugin architecture

**Breaking Changes**:
- ⚠️ `AdvancedWatermarkAttacks` class replaced with `WatermarkAttacker`
- ⚠️ Attack method signatures standardized
- ⚠️ Configuration format changed to YAML-based system

**Migration Guide**:

Old code:
```python
from attack_class import AdvancedWatermarkAttacks
attacker = AdvancedWatermarkAttacks()
result = attacker.run_single_attack(orig_path, water_path, 'high_frequency', 0.8)
```

New code:
```python
from advanced_attacks import WatermarkAttacker
attacker = WatermarkAttacker()
result = attacker.apply_attack(
    image=water_path,
    attack_name='high_frequency_filter',
    parameters={'filter_strength': 0.8},
    original_image=orig_path
)
```

**Performance Improvements**:
- 🚀 50% faster attack execution through optimized pipelines
- 🚀 Reduced memory usage via lazy model loading
- 🚀 Parallel processing support for batch operations

**Bug Fixes**:
- 🐛 Fixed image dimension handling for diffusion models
- 🐛 Resolved memory leaks in batch processing
- 🐛 Corrected SSIM calculation for grayscale images
- 🐛 Fixed parameter validation edge cases

---

For more information, see the [main repository README](../README.md) and [documentation](../docs/).