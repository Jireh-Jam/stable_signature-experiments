# Transformations Pipeline

Comprehensive image transformation system for watermark robustness testing and data augmentation.

## Overview

The transformations pipeline provides a unified, extensible framework for applying image transformations that simulate real-world conditions and adversarial attacks. It combines multiple transformation libraries and provides a registry-based system for easy discovery and configuration.

## Concept & Purpose

**Transformations** in this context refer to image modifications that can:
- **Simulate** real-world image processing (compression, resizing, filtering)
- **Test** watermark robustness against common operations
- **Generate** adversarial examples for security evaluation
- **Augment** training data for improved model robustness
- **Evaluate** perceptual quality under various distortions

The pipeline is designed to be **deterministic** (with seeding), **composable** (chain multiple transforms), and **configurable** (parameter sweeps and optimization).

## Pipeline Location & Architecture

### Code Organization

```
common/
├── transforms.py           # Main pipeline implementation
├── image_utils.py         # Image I/O and utility functions
└── config.py             # Configuration management

tools/
└── transformations.py     # Legacy transform utilities (consolidated)

combined_transforms.py     # Legacy combined transforms (consolidated)
```

### Pipeline Anatomy

```
Input Image → Transform Registry → Parameter Resolution → Transform Chain → Output Image
     ↓              ↓                    ↓                    ↓              ↓
  PIL Image    Available Transforms   Default/Custom      Sequential      Modified
               & Metadata            Parameters          Application      Image
```

## Registry Pattern

The pipeline uses a **registry pattern** for transform management:

```python
from common.transforms import transform_registry

# List available transforms
transforms = transform_registry.list_transforms()
print(f"Available: {transforms}")

# Get specific transform
blur_transform = transform_registry.get_transform('gaussian_blur')

# Apply transform
result = transform_registry.apply_transform(
    image=my_image,
    transform_name='gaussian_blur',
    radius=2.0
)
```

### Transform Categories

| Category | Transforms | Purpose |
|----------|------------|---------|
| **Geometric** | `crop_center`, `crop_random`, `resize`, `rotate`, `perspective` | Spatial modifications |
| **Filtering** | `gaussian_blur`, `motion_blur`, `sharpen` | Frequency domain changes |
| **Color** | `brightness`, `contrast`, `saturation`, `hue`, `gamma`, `grayscale` | Color space modifications |
| **Noise** | `gaussian_noise`, `salt_pepper_noise`, `random_erasing` | Additive distortions |
| **Compression** | `jpeg_compression`, `bit_mask` | Lossy encoding effects |
| **Overlay** | `text_overlay` | Content additions |

## Transform Ordering & Composition

### Sequential Application

Transforms are applied in the order specified:

```python
# Define transform chain
transform_chain = [
    ('resize', {'scale': 0.8}),
    ('gaussian_blur', {'radius': 1.5}),
    ('jpeg_compression', {'quality': 70}),
    ('brightness', {'factor': 1.2})
]

# Apply chain
results = transform_registry.apply_transform_chain(image, transform_chain)

# Check results
for i, result in enumerate(results):
    if result.success:
        print(f"Transform {i+1}: Success")
    else:
        print(f"Transform {i+1}: Failed - {result.error_message}")
```

### Composition Strategies

**1. Additive Composition** (independent transforms):
```python
# Apply each transform to original image
base_transforms = ['gaussian_blur', 'brightness', 'contrast']
results = {}

for transform_name in base_transforms:
    result = transform_registry.apply_transform(original_image, transform_name)
    results[transform_name] = result
```

**2. Sequential Composition** (cumulative effects):
```python
# Each transform builds on previous result
current_image = original_image
transform_sequence = [
    ('resize', {'scale': 0.9}),
    ('rotate', {'degrees': 5}),
    ('gaussian_blur', {'radius': 1.0})
]

for transform_name, params in transform_sequence:
    result = transform_registry.apply_transform(current_image, transform_name, **params)
    if result.success:
        current_image = result.image
```

**3. Parallel Composition** (multiple variants):
```python
# Generate multiple versions simultaneously
blur_variants = [
    ('gaussian_blur', {'radius': 0.5}),
    ('gaussian_blur', {'radius': 1.0}),
    ('gaussian_blur', {'radius': 2.0}),
    ('motion_blur', {'size': 5})
]

variant_images = []
for transform_name, params in blur_variants:
    result = transform_registry.apply_transform(original_image, transform_name, **params)
    if result.success:
        variant_images.append(result.image)
```

## Parameterization & Configuration

### Default Parameters

Each transform has sensible defaults:

```python
# Get default parameters for a transform
defaults = transform_registry.get_transform('gaussian_blur').get_default_parameters()
print(defaults)  # {'radius': 1.0}

# Apply with defaults
result = transform_registry.apply_transform(image, 'gaussian_blur')
```

### Parameter Ranges

Standard configurations provide parameter sweeps:

```python
from common.transforms import get_standard_transform_configs

configs = get_standard_transform_configs()
print(configs['blur_light'])    # {'radius': 1.0}
print(configs['blur_medium'])   # {'radius': 2.0}
print(configs['blur_heavy'])    # {'radius': 3.0}
```

### Custom Parameters

Override defaults with custom values:

```python
# Custom blur with specific radius
result = transform_registry.apply_transform(
    image=my_image,
    transform_name='gaussian_blur',
    radius=2.5  # Custom value
)

# Custom JPEG compression
result = transform_registry.apply_transform(
    image=my_image,
    transform_name='jpeg_compression',
    quality=45  # Lower quality than default
)
```

## Determinism & Seeding

### Reproducible Transforms

Set seeds for deterministic behavior:

```python
import random
import numpy as np
import torch

# Set all seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Apply random transforms
result1 = transform_registry.apply_transform(image, 'crop_random', crop_percentage=0.1)
result2 = transform_registry.apply_transform(image, 'crop_random', crop_percentage=0.1)

# Results will be identical due to seeding
assert np.array_equal(np.array(result1.image), np.array(result2.image))
```

### Per-Transform Seeding

```python
# Seed individual transforms
class SeededRandomCrop(BaseTransform):
    def apply(self, image, crop_percentage=0.1, seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Apply random crop with seeded randomness
        # ... implementation
        
        return TransformResult(image=cropped_image, parameters={'seed': seed}, success=True)
```

## Device Placement

### CPU vs GPU Processing

Most transforms run on CPU (PIL/NumPy-based):

```python
# CPU-based transforms (default)
result = transform_registry.apply_transform(image, 'gaussian_blur')

# For GPU-accelerated transforms (custom implementation)
class GPUTransform(BaseTransform):
    def __init__(self, device='cuda'):
        super().__init__('gpu_transform')
        self.device = device
    
    def apply(self, image, **kwargs):
        # Convert to tensor
        tensor = transforms.ToTensor()(image).to(self.device)
        
        # Apply GPU operations
        processed_tensor = self.gpu_operation(tensor)
        
        # Convert back to PIL
        result_image = transforms.ToPILImage()(processed_tensor.cpu())
        
        return TransformResult(image=result_image, parameters=kwargs, success=True)
```

## Adding New Transforms

### Step-by-Step Template

**1. Create Transform Class**:

```python
from common.transforms import BaseTransform, TransformResult
from PIL import Image, ImageFilter
import numpy as np

class MyCustomTransform(BaseTransform):
    def __init__(self):
        super().__init__(
            name='my_custom_transform',
            description='Description of what this transform does'
        )
    
    def apply(self, image: Image.Image, **kwargs) -> TransformResult:
        """
        Apply custom transformation to image.
        
        Args:
            image: Input PIL Image
            **kwargs: Transform parameters
            
        Returns:
            TransformResult with processed image
        """
        try:
            # Extract parameters with defaults
            strength = kwargs.get('strength', 1.0)
            mode = kwargs.get('mode', 'default')
            
            # Validate parameters
            if not 0 <= strength <= 2.0:
                raise ValueError(f"strength must be in [0, 2.0], got {strength}")
            
            # Apply transformation
            processed_image = self._apply_custom_operation(image, strength, mode)
            
            return TransformResult(
                image=processed_image,
                parameters={'strength': strength, 'mode': mode},
                success=True
            )
            
        except Exception as e:
            return TransformResult(
                image=image,  # Return original on failure
                parameters=kwargs,
                success=False,
                error_message=str(e)
            )
    
    def _apply_custom_operation(self, image, strength, mode):
        """Implement your custom image operation here."""
        # Example: Custom blur with strength parameter
        radius = strength * 2.0
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def get_default_parameters(self):
        """Return default parameters for this transform."""
        return {
            'strength': 1.0,
            'mode': 'default'
        }
    
    def validate_parameters(self, **kwargs):
        """Validate transform parameters."""
        strength = kwargs.get('strength', 1.0)
        mode = kwargs.get('mode', 'default')
        
        if not isinstance(strength, (int, float)) or not 0 <= strength <= 2.0:
            return False
        
        if mode not in ['default', 'aggressive', 'gentle']:
            return False
        
        return True
```

**2. Register Transform**:

```python
from common.transforms import transform_registry

# Create and register transform
my_transform = MyCustomTransform()
transform_registry.register('my_custom_transform', my_transform)

# Verify registration
assert 'my_custom_transform' in transform_registry.list_transforms()
```

**3. Add Type Hints & Docstring Requirements**:

```python
from typing import Dict, Any, Union
from PIL import Image

class MyCustomTransform(BaseTransform):
    def apply(self, image: Image.Image, **kwargs: Any) -> TransformResult:
        """
        Apply custom transformation with proper documentation.
        
        This transform applies a custom operation that combines blur and 
        brightness adjustment based on the strength parameter.
        
        Args:
            image: Input PIL Image in RGB format
            strength: Transform strength (0.0 to 2.0, default: 1.0)
                     0.0 = no effect, 1.0 = normal, 2.0 = maximum
            mode: Processing mode ('default', 'aggressive', 'gentle')
                  Controls how the transform is applied
            
        Returns:
            TransformResult containing:
                - image: Processed PIL Image
                - parameters: Applied parameters
                - success: True if successful, False otherwise
                - error_message: Error description if failed
                
        Raises:
            ValueError: If parameters are outside valid ranges
            
        Example:
            >>> transform = MyCustomTransform()
            >>> result = transform.apply(image, strength=1.5, mode='aggressive')
            >>> if result.success:
            ...     processed_image = result.image
        """
        # Implementation here...
```

**4. Test Transform**:

```python
# Unit test template
def test_my_custom_transform():
    from PIL import Image
    import numpy as np
    
    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    # Test with default parameters
    transform = MyCustomTransform()
    result = transform.apply(test_image)
    
    assert result.success
    assert result.image.size == test_image.size
    assert 'strength' in result.parameters
    
    # Test with custom parameters
    result = transform.apply(test_image, strength=1.5, mode='aggressive')
    assert result.success
    assert result.parameters['strength'] == 1.5
    assert result.parameters['mode'] == 'aggressive'
    
    # Test parameter validation
    result = transform.apply(test_image, strength=5.0)  # Invalid
    assert not result.success
    assert 'strength' in result.error_message
    
    print("All tests passed!")

# Run test
test_my_custom_transform()
```

## Example Pipelines

### Simple Pipeline

```python
from common.transforms import transform_registry
from PIL import Image

# Load image
image = Image.open('test.jpg')

# Apply single transform
result = transform_registry.apply_transform(
    image=image,
    transform_name='gaussian_blur',
    radius=1.5
)

if result.success:
    result.image.save('blurred.jpg')
```

### Medium Complexity Pipeline

```python
# Multi-step processing pipeline
def create_robustness_variants(image_path, output_dir):
    """Create multiple variants for robustness testing."""
    
    image = Image.open(image_path)
    variants = {}
    
    # Define transform configurations
    transform_configs = {
        'light_blur': ('gaussian_blur', {'radius': 1.0}),
        'medium_blur': ('gaussian_blur', {'radius': 2.0}),
        'jpeg_90': ('jpeg_compression', {'quality': 90}),
        'jpeg_70': ('jpeg_compression', {'quality': 70}),
        'resize_80': ('resize', {'scale': 0.8}),
        'bright_120': ('brightness', {'factor': 1.2}),
        'contrast_80': ('contrast', {'factor': 0.8})
    }
    
    # Apply each transform
    for variant_name, (transform_name, params) in transform_configs.items():
        result = transform_registry.apply_transform(image, transform_name, **params)
        
        if result.success:
            output_path = Path(output_dir) / f'{variant_name}.jpg'
            result.image.save(output_path)
            variants[variant_name] = {
                'path': output_path,
                'parameters': result.parameters
            }
        else:
            print(f"Failed to create variant {variant_name}: {result.error_message}")
    
    return variants

# Usage
variants = create_robustness_variants('input.jpg', 'variants/')
print(f"Created {len(variants)} variants")
```

### Advanced Pipeline

```python
from itertools import product
import json

def comprehensive_transform_sweep(image_path, output_dir):
    """
    Generate comprehensive transform combinations for research.
    """
    
    image = Image.open(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define parameter grids
    blur_radii = [0.5, 1.0, 1.5, 2.0]
    jpeg_qualities = [50, 70, 85, 95]
    brightness_factors = [0.8, 0.9, 1.1, 1.2]
    
    results = []
    
    # Single transforms
    for radius in blur_radii:
        result = transform_registry.apply_transform(image, 'gaussian_blur', radius=radius)
        if result.success:
            filename = f'blur_r{radius}.jpg'
            result.image.save(output_dir / filename)
            results.append({
                'filename': filename,
                'transforms': [('gaussian_blur', {'radius': radius})],
                'success': True
            })
    
    # Combined transforms
    for quality, brightness in product(jpeg_qualities, brightness_factors):
        # Apply JPEG then brightness
        chain = [
            ('jpeg_compression', {'quality': quality}),
            ('brightness', {'factor': brightness})
        ]
        
        chain_results = transform_registry.apply_transform_chain(image, chain)
        
        if all(r.success for r in chain_results):
            final_image = chain_results[-1].image
            filename = f'jpeg{quality}_bright{brightness}.jpg'
            final_image.save(output_dir / filename)
            
            results.append({
                'filename': filename,
                'transforms': chain,
                'success': True
            })
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Usage
results = comprehensive_transform_sweep('input.jpg', 'sweep_results/')
print(f"Generated {len(results)} transform combinations")
```

## Config-Driven Usage

### YAML Configuration

Create `transforms_config.yaml`:

```yaml
transforms:
  basic_robustness:
    - name: gaussian_blur
      parameters:
        radius: [0.5, 1.0, 1.5, 2.0]
    
    - name: jpeg_compression
      parameters:
        quality: [50, 70, 85, 95]
    
    - name: brightness
      parameters:
        factor: [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]

  geometric_attacks:
    - name: resize
      parameters:
        scale: [0.5, 0.7, 0.8, 0.9]
    
    - name: rotate
      parameters:
        degrees: [-10, -5, 5, 10, 15]
    
    - name: crop_center
      parameters:
        crop_percentage: [0.1, 0.2, 0.3]

  combined_attacks:
    - chain:
        - name: resize
          parameters: {scale: 0.8}
        - name: gaussian_blur
          parameters: {radius: 1.0}
        - name: jpeg_compression
          parameters: {quality: 70}
```

### Configuration Parser

```python
import yaml
from pathlib import Path
from itertools import product

def load_transform_config(config_path):
    """Load transform configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def apply_config_transforms(image, config, category='basic_robustness'):
    """Apply transforms based on configuration."""
    
    transforms_config = config['transforms'][category]
    results = {}
    
    for transform_config in transforms_config:
        if 'chain' in transform_config:
            # Handle transform chains
            chain = [(t['name'], t['parameters']) for t in transform_config['chain']]
            chain_results = transform_registry.apply_transform_chain(image, chain)
            
            if all(r.success for r in chain_results):
                chain_name = '_'.join([t['name'] for t in transform_config['chain']])
                results[f'chain_{chain_name}'] = chain_results[-1].image
        
        else:
            # Handle single transforms with parameter grids
            transform_name = transform_config['name']
            parameters = transform_config['parameters']
            
            # Generate all parameter combinations
            param_names = list(parameters.keys())
            param_values = list(parameters.values())
            
            for param_combo in product(*param_values):
                param_dict = dict(zip(param_names, param_combo))
                
                result = transform_registry.apply_transform(
                    image, transform_name, **param_dict
                )
                
                if result.success:
                    # Create descriptive name
                    param_str = '_'.join([f'{k}{v}' for k, v in param_dict.items()])
                    result_name = f'{transform_name}_{param_str}'
                    results[result_name] = result.image
    
    return results

# Usage
config = load_transform_config('transforms_config.yaml')
image = Image.open('test.jpg')

# Apply basic robustness transforms
basic_results = apply_config_transforms(image, config, 'basic_robustness')
print(f"Generated {len(basic_results)} basic variants")

# Apply geometric attacks
geometric_results = apply_config_transforms(image, config, 'geometric_attacks')
print(f"Generated {len(geometric_results)} geometric variants")
```

## Testing Transforms

### Quick Sanity Check

```python
def test_transform_sanity(transform_name, image_path='test.jpg'):
    """Quick sanity check for a transform."""
    
    # Load test image
    image = Image.open(image_path)
    original_size = image.size
    
    # Apply transform with defaults
    result = transform_registry.apply_transform(image, transform_name)
    
    # Basic checks
    assert result.success, f"Transform failed: {result.error_message}"
    assert isinstance(result.image, Image.Image), "Result is not PIL Image"
    assert result.image.mode == image.mode, "Image mode changed unexpectedly"
    
    # Size check (may change for some transforms)
    if transform_name not in ['resize', 'crop_center', 'crop_random', 'rotate']:
        assert result.image.size == original_size, f"Size changed: {original_size} -> {result.image.size}"
    
    print(f"✓ Transform '{transform_name}' passed sanity check")
    return True

# Test all transforms
for transform_name in transform_registry.list_transforms():
    try:
        test_transform_sanity(transform_name)
    except Exception as e:
        print(f"✗ Transform '{transform_name}' failed: {e}")
```

### Performance Benchmarking

```python
import time
import statistics

def benchmark_transform(transform_name, image_path='test.jpg', num_runs=10):
    """Benchmark transform performance."""
    
    image = Image.open(image_path)
    times = []
    
    # Warm-up run
    transform_registry.apply_transform(image, transform_name)
    
    # Benchmark runs
    for _ in range(num_runs):
        start_time = time.time()
        result = transform_registry.apply_transform(image, transform_name)
        end_time = time.time()
        
        if result.success:
            times.append(end_time - start_time)
    
    if times:
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"Transform '{transform_name}':")
        print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Min time: {min(times):.3f}s")
        print(f"  Max time: {max(times):.3f}s")
        
        return avg_time
    else:
        print(f"Transform '{transform_name}' failed all runs")
        return None

# Benchmark all transforms
print("Transform Performance Benchmark")
print("=" * 40)

for transform_name in sorted(transform_registry.list_transforms()):
    benchmark_transform(transform_name)
    print()
```

## Performance Tips

### Optimization Strategies

**1. Batch Processing**:
```python
# Process multiple images efficiently
def batch_transform(image_paths, transform_name, **params):
    """Apply same transform to multiple images."""
    results = []
    
    for image_path in image_paths:
        image = Image.open(image_path)
        result = transform_registry.apply_transform(image, transform_name, **params)
        results.append((image_path, result))
    
    return results
```

**2. Lazy Loading**:
```python
# Only load images when needed
class LazyImageTransform:
    def __init__(self, image_path):
        self.image_path = image_path
        self._image = None
    
    @property
    def image(self):
        if self._image is None:
            self._image = Image.open(self.image_path)
        return self._image
    
    def apply_transform(self, transform_name, **params):
        return transform_registry.apply_transform(self.image, transform_name, **params)
```

**3. Memory Management**:
```python
# Clear memory between operations
import gc

def memory_efficient_processing(image_paths, transforms):
    """Process with memory cleanup."""
    
    for image_path in image_paths:
        image = Image.open(image_path)
        
        for transform_name, params in transforms:
            result = transform_registry.apply_transform(image, transform_name, **params)
            
            if result.success:
                # Save result immediately
                output_path = f"{image_path.stem}_{transform_name}.jpg"
                result.image.save(output_path)
        
        # Clean up
        del image
        gc.collect()
```

### Vectorization

```python
# Use NumPy for faster operations where possible
import numpy as np

class FastNoiseTransform(BaseTransform):
    def apply(self, image, noise_level=0.1):
        # Convert to numpy for vectorized operations
        img_array = np.array(image, dtype=np.float32)
        
        # Vectorized noise addition
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert back to PIL
        result_image = Image.fromarray(noisy_array)
        
        return TransformResult(
            image=result_image,
            parameters={'noise_level': noise_level},
            success=True
        )
```

## Common Pitfalls

### Image Format Issues

```python
# Handle different image formats properly
def safe_transform_apply(image, transform_name, **params):
    """Apply transform with format safety."""
    
    # Ensure RGB format
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            # Handle transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        else:
            image = image.convert('RGB')
    
    # Apply transform
    result = transform_registry.apply_transform(image, transform_name, **params)
    
    return result
```

### Parameter Validation

```python
# Always validate parameters
class RobustTransform(BaseTransform):
    def apply(self, image, strength=1.0, **kwargs):
        # Type checking
        if not isinstance(strength, (int, float)):
            raise TypeError(f"strength must be numeric, got {type(strength)}")
        
        # Range checking
        if not 0 <= strength <= 2.0:
            raise ValueError(f"strength must be in [0, 2.0], got {strength}")
        
        # Apply transform...
```

### Error Handling

```python
# Graceful error handling
def robust_transform_chain(image, transform_chain):
    """Apply transform chain with error recovery."""
    
    current_image = image
    successful_transforms = []
    
    for transform_name, params in transform_chain:
        try:
            result = transform_registry.apply_transform(current_image, transform_name, **params)
            
            if result.success:
                current_image = result.image
                successful_transforms.append((transform_name, params))
            else:
                print(f"Transform {transform_name} failed: {result.error_message}")
                # Continue with previous image
        
        except Exception as e:
            print(f"Exception in transform {transform_name}: {e}")
            # Continue with previous image
    
    return current_image, successful_transforms
```

---

For more information, see the [main repository README](../README.md) and [advanced attacks documentation](../advanced_attacks/README.md).