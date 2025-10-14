# Transformations Pipeline Documentation

## Concept

The transformations pipeline provides a flexible system for applying image augmentations and modifications. These transformations serve multiple purposes:

1. **Data Augmentation**: Training robust watermarking models
2. **Attack Simulation**: Testing watermark resilience
3. **Preprocessing**: Preparing images for model input
4. **Evaluation**: Systematic testing of watermark robustness

Transformations can be composed, parameterized, and applied deterministically or stochastically.

## Pipeline Anatomy

### Core Components

```
Transformation Pipeline
├── Base Transformations      # Individual operations
├── Transform Registry        # Catalog of available transforms
├── Transform Composers       # Combine multiple transforms
├── Parameter Controllers     # Dynamic parameter adjustment
└── Execution Engine         # Apply transforms with logging
```

### Code Locations

- **`tools/transformations.py`**: Main transformation implementations
- **`combined_transforms.py`**: PyTorch-based transformations with normalization
- **`common/image_utils.py`**: Shared image processing utilities
- **Attack modules**: Specialized transformation usage

### Transform Categories

1. **Geometric Transforms**
   - Rotation, scaling, cropping, perspective

2. **Color/Intensity Transforms**
   - Brightness, contrast, saturation, hue adjustment

3. **Noise and Artifacts**
   - Gaussian noise, compression artifacts, blur

4. **Frequency Domain**
   - High-frequency filtering, bandpass operations

5. **AI-Based Transforms**
   - Diffusion model regeneration, neural compression

## Registry Pattern

The transformation system uses a registry pattern for flexibility:

```python
class TransformRegistry:
    """Central registry for all transformations."""
    
    _transforms = {}
    
    @classmethod
    def register(cls, name: str, transform_func: Callable):
        """Register a transformation."""
        cls._transforms[name] = transform_func
    
    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a transformation by name."""
        return cls._transforms.get(name)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered transformations."""
        return list(cls._transforms.keys())
```

## Transform Ordering and Composition

### Sequential Composition

```python
def compose_transforms(transforms: List[Callable]) -> Callable:
    """Compose multiple transforms sequentially."""
    def composed(image):
        result = image
        for transform in transforms:
            result = transform(result)
        return result
    return composed

# Example usage
pipeline = compose_transforms([
    lambda img: ImageTransformations.gaussian_blur(img, radius=1.0),
    lambda img: ImageTransformations.adjust_brightness(img, factor=1.1),
    lambda img: ImageTransformations.jpeg_compression(img, quality=85)
])
```

### Conditional Composition

```python
def conditional_transform(condition: Callable, transform: Callable) -> Callable:
    """Apply transform only if condition is met."""
    def conditional(image):
        if condition(image):
            return transform(image)
        return image
    return conditional

# Apply blur only to bright images
blur_if_bright = conditional_transform(
    lambda img: np.mean(img) > 128,
    lambda img: ImageTransformations.gaussian_blur(img, 2.0)
)
```

### Probabilistic Composition

```python
def random_transform(transforms: List[Tuple[Callable, float]]) -> Callable:
    """Apply transforms with specified probabilities."""
    def randomized(image):
        for transform, prob in transforms:
            if random.random() < prob:
                image = transform(image)
        return image
    return randomized

# 50% chance of blur, 30% chance of noise
augmentation = random_transform([
    (lambda img: ImageTransformations.gaussian_blur(img, 1.5), 0.5),
    (lambda img: ImageTransformations.add_noise(img, 0.05), 0.3)
])
```

## Parameterization

### Static Parameters

```python
# Fixed parameters
blur_transform = lambda img: ImageTransformations.gaussian_blur(img, radius=2.0)
```

### Dynamic Parameters

```python
class DynamicTransform:
    """Transform with adjustable parameters."""
    
    def __init__(self, base_transform: Callable, param_ranges: Dict):
        self.base_transform = base_transform
        self.param_ranges = param_ranges
    
    def __call__(self, image, **override_params):
        # Sample or use provided parameters
        params = {}
        for name, (min_val, max_val) in self.param_ranges.items():
            if name in override_params:
                params[name] = override_params[name]
            else:
                params[name] = random.uniform(min_val, max_val)
        
        return self.base_transform(image, **params)

# Dynamic blur with radius 0.5-3.0
dynamic_blur = DynamicTransform(
    ImageTransformations.gaussian_blur,
    {'radius': (0.5, 3.0)}
)
```

### Configuration-Driven Parameters

```python
# Load from config file
import json

def load_transform_config(config_path: str) -> Dict:
    """Load transformation configuration."""
    with open(config_path) as f:
        return json.load(f)

# Config example (transform_config.json):
{
    "pipeline": [
        {
            "name": "gaussian_blur",
            "params": {"radius": 1.5},
            "probability": 0.8
        },
        {
            "name": "brightness",
            "params": {"factor": 1.2},
            "probability": 0.5
        }
    ]
}

# Build pipeline from config
def build_pipeline_from_config(config: Dict) -> Callable:
    transforms = []
    for item in config['pipeline']:
        transform = TransformRegistry.get(item['name'])
        params = item['params']
        prob = item.get('probability', 1.0)
        
        # Create parameterized transform
        parameterized = lambda img, p=params: transform(img, **p)
        transforms.append((parameterized, prob))
    
    return random_transform(transforms)
```

## Determinism and Seeding

### Ensuring Reproducibility

```python
import random
import numpy as np

class DeterministicPipeline:
    """Pipeline with controlled randomness."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
    
    def reset(self):
        """Reset to initial seed."""
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)
    
    def apply_transform(self, image, transform):
        """Apply transform with controlled randomness."""
        # Save current RNG state
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        
        # Set deterministic state
        random.setstate(self.rng.getstate())
        np.random.set_state(self.np_rng.get_state())
        
        # Apply transform
        result = transform(image)
        
        # Update internal state
        self.rng.setstate(random.getstate())
        self.np_rng.set_state(np.random.get_state())
        
        # Restore original state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        
        return result
```

### Device Placement

```python
def ensure_device(transform: Callable, device: str) -> Callable:
    """Ensure transform operates on specified device."""
    def device_aware(image):
        if isinstance(image, torch.Tensor):
            image = image.to(device)
        
        result = transform(image)
        
        if isinstance(result, torch.Tensor):
            result = result.to(device)
        
        return result
    return device_aware
```

## Adding New Transforms

### Step 1: Define Transform Function

```python
# In tools/transformations.py or custom module
@staticmethod
def swirl_distortion(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """
    Apply swirl distortion to image.
    
    Args:
        image: Input PIL Image
        strength: Distortion strength (0-1)
        
    Returns:
        Distorted image
    """
    # Implementation using scipy or custom algorithm
    import numpy as np
    from scipy.ndimage import map_coordinates
    
    # Convert to array
    arr = np.array(image)
    rows, cols = arr.shape[:2]
    
    # Create swirl mapping
    center = (rows/2, cols/2)
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center
    dy = y - center[0]
    dx = x - center[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Apply swirl
    angle = strength * np.exp(-distance / (max(rows, cols) * 0.5))
    y_new = dy * np.cos(angle) - dx * np.sin(angle) + center[0]
    x_new = dy * np.sin(angle) + dx * np.cos(angle) + center[1]
    
    # Apply transformation
    if len(arr.shape) == 3:
        # Color image
        result = np.zeros_like(arr)
        for i in range(arr.shape[2]):
            result[:,:,i] = map_coordinates(arr[:,:,i], [y_new, x_new])
    else:
        # Grayscale
        result = map_coordinates(arr, [y_new, x_new])
    
    return Image.fromarray(result.astype(np.uint8))
```

### Step 2: Add Type Hints and Validation

```python
from typing import Union, Tuple
import numpy as np
from PIL import Image

def validate_image_input(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Validate and convert image input."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image
    else:
        raise TypeError(f"Expected PIL Image or numpy array, got {type(image)}")

def swirl_distortion(
    image: Union[Image.Image, np.ndarray], 
    strength: float = 0.5,
    center: Optional[Tuple[float, float]] = None
) -> Image.Image:
    """Apply swirl distortion with validation."""
    image = validate_image_input(image)
    
    if not 0 <= strength <= 1:
        raise ValueError(f"Strength must be in [0, 1], got {strength}")
    
    # ... implementation ...
```

### Step 3: Register Transform

```python
# In module initialization or registry
TransformRegistry.register('swirl', swirl_distortion)

# Or in class method
ImageTransformations.swirl = swirl_distortion
```

### Step 4: Add to Standard Suites

```python
@classmethod
def get_extended_transformations(cls) -> Dict[str, Callable]:
    """Get extended transformation suite including new transforms."""
    transforms = cls.get_standard_transformations()
    transforms.update({
        'swirl_light': lambda img: cls.swirl_distortion(img, 0.3),
        'swirl_heavy': lambda img: cls.swirl_distortion(img, 0.7),
    })
    return transforms
```

### Step 5: Document

```python
def swirl_distortion(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """
    Apply swirl distortion to image.
    
    This transformation creates a swirling effect centered on the image,
    useful for testing watermark robustness to non-linear geometric distortions.
    
    Args:
        image: Input PIL Image
        strength: Distortion strength (0-1)
            - 0: No distortion
            - 0.3: Light swirl
            - 0.5: Medium swirl (default)
            - 0.7: Heavy swirl
            - 1.0: Maximum swirl
        
    Returns:
        Distorted PIL Image
        
    Example:
        >>> img = Image.open('input.jpg')
        >>> swirled = swirl_distortion(img, strength=0.5)
        >>> swirled.save('output.jpg')
    """
```

## Example Pipelines

### Simple Pipeline

```python
# Basic watermark robustness test
simple_pipeline = compose_transforms([
    lambda img: ImageTransformations.gaussian_blur(img, 1.0),
    lambda img: ImageTransformations.jpeg_compression(img, 85),
    lambda img: ImageTransformations.adjust_brightness(img, 1.1)
])

result = simple_pipeline(input_image)
```

### Medium Complexity Pipeline

```python
# Realistic social media processing simulation
social_media_pipeline = compose_transforms([
    # Resize for upload
    lambda img: ImageTransformations.resize_image(img, 0.8),
    
    # Random filter (50% chance)
    random_transform([
        (lambda img: ImageTransformations.adjust_saturation(img, 1.2), 0.5)
    ]),
    
    # Compression
    lambda img: ImageTransformations.jpeg_compression(img, 75),
    
    # Platform-specific processing
    lambda img: ImageTransformations.adjust_contrast(img, 1.05)
])
```

### Advanced Pipeline

```python
class AdvancedPipeline:
    """Complex transformation pipeline with state tracking."""
    
    def __init__(self, config_path: str):
        self.config = load_transform_config(config_path)
        self.history = []
        self.metrics = {}
    
    def apply(self, image: Image.Image) -> Image.Image:
        """Apply pipeline with tracking."""
        current = image
        self.history = []
        
        for step in self.config['pipeline']:
            # Record state
            self.history.append({
                'transform': step['name'],
                'params': step['params'],
                'input_hash': hash(current.tobytes())
            })
            
            # Apply transform
            transform = TransformRegistry.get(step['name'])
            current = transform(current, **step['params'])
            
            # Calculate metrics
            if 'calculate_metrics' in step and step['calculate_metrics']:
                self.metrics[step['name']] = self._calculate_metrics(image, current)
        
        return current
    
    def _calculate_metrics(self, original: Image.Image, transformed: Image.Image) -> Dict:
        """Calculate quality metrics."""
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
        
        orig_array = np.array(original)
        trans_array = np.array(transformed)
        
        # PSNR
        mse = np.mean((orig_array - trans_array) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # SSIM
        ssim_value = ssim(orig_array, trans_array, multichannel=True)
        
        return {
            'psnr': psnr,
            'ssim': ssim_value,
            'mse': mse
        }
    
    def get_report(self) -> Dict:
        """Get execution report."""
        return {
            'history': self.history,
            'metrics': self.metrics,
            'final_metrics': self.metrics.get(self.history[-1]['transform'], {})
        }
```

## Config-Driven Usage

### Sample Configuration

```yaml
# transform_pipeline.yaml
pipeline:
  name: "robustness_test_v1"
  description: "Standard robustness testing pipeline"
  version: "1.0"
  
stages:
  - name: "preprocessing"
    transforms:
      - type: "resize"
        params:
          scale: 1.0
        required: true
  
  - name: "augmentation"
    transforms:
      - type: "gaussian_blur"
        params:
          radius: !range [0.5, 2.0]
        probability: 0.7
      
      - type: "gaussian_noise"
        params:
          std: !range [0.01, 0.05]
        probability: 0.5
  
  - name: "compression"
    transforms:
      - type: "jpeg_compression"
        params:
          quality: !choice [70, 80, 90]
        required: true

output:
  save_intermediate: true
  metrics: ["psnr", "ssim", "bit_accuracy"]
```

### Parser Implementation

```python
import yaml
from typing import Any, Dict

class PipelineConfig:
    """Parse and validate pipeline configuration."""
    
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self._validate()
    
    def _validate(self):
        """Validate configuration structure."""
        required = ['pipeline', 'stages']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required key: {key}")
    
    def build_pipeline(self) -> Callable:
        """Build executable pipeline from config."""
        transforms = []
        
        for stage in self.config['stages']:
            for transform in stage['transforms']:
                # Get transform function
                func = TransformRegistry.get(transform['type'])
                
                # Handle parameter specifications
                params = self._resolve_params(transform['params'])
                
                # Add to pipeline
                if transform.get('required', False):
                    transforms.append(lambda img, p=params: func(img, **p))
                else:
                    prob = transform.get('probability', 1.0)
                    transforms.append((
                        lambda img, p=params: func(img, **p),
                        prob
                    ))
        
        return self._combine_transforms(transforms)
    
    def _resolve_params(self, params: Dict) -> Dict:
        """Resolve parameter specifications."""
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, dict):
                if 'range' in value:
                    # Sample from range
                    min_val, max_val = value['range']
                    resolved[key] = random.uniform(min_val, max_val)
                elif 'choice' in value:
                    # Sample from choices
                    resolved[key] = random.choice(value['choice'])
            else:
                resolved[key] = value
        
        return resolved
```

## Testing Transforms

### Unit Testing

```python
import unittest
import numpy as np
from PIL import Image

class TestTransformations(unittest.TestCase):
    """Test transformation correctness."""
    
    def setUp(self):
        """Create test image."""
        self.test_image = Image.new('RGB', (100, 100), color='red')
    
    def test_blur_reduces_sharpness(self):
        """Test that blur reduces image sharpness."""
        blurred = ImageTransformations.gaussian_blur(self.test_image, 2.0)
        
        # Convert to arrays
        orig_array = np.array(self.test_image)
        blur_array = np.array(blurred)
        
        # Calculate gradient magnitude (sharpness metric)
        orig_grad = np.gradient(orig_array.mean(axis=2))
        blur_grad = np.gradient(blur_array.mean(axis=2))
        
        orig_sharpness = np.mean(np.abs(orig_grad[0]) + np.abs(orig_grad[1]))
        blur_sharpness = np.mean(np.abs(blur_grad[0]) + np.abs(blur_grad[1]))
        
        self.assertLess(blur_sharpness, orig_sharpness)
    
    def test_transform_preserves_size(self):
        """Test that transforms preserve image dimensions."""
        transforms = [
            lambda img: ImageTransformations.gaussian_blur(img, 1.0),
            lambda img: ImageTransformations.adjust_brightness(img, 1.2),
            lambda img: ImageTransformations.add_noise(img, 0.05)
        ]
        
        for transform in transforms:
            result = transform(self.test_image)
            self.assertEqual(result.size, self.test_image.size)
    
    def test_deterministic_with_seed(self):
        """Test deterministic behavior with seeding."""
        pipeline = DeterministicPipeline(seed=42)
        
        # Apply twice with same seed
        result1 = pipeline.apply_transform(
            self.test_image,
            lambda img: ImageTransformations.add_noise(img, 0.1)
        )
        
        pipeline.reset()
        result2 = pipeline.apply_transform(
            self.test_image,
            lambda img: ImageTransformations.add_noise(img, 0.1)
        )
        
        # Should be identical
        np.testing.assert_array_equal(np.array(result1), np.array(result2))
```

### Integration Testing

```python
def test_pipeline_integration():
    """Test full pipeline with watermark detection."""
    from detector import WatermarkDetector
    
    # Setup
    detector = WatermarkDetector("ckpts/hidden_replicate.pth")
    pipeline = load_standard_pipeline()
    
    # Original watermarked image
    watermarked = Image.open("test_watermarked.png")
    original_msg = detector.detect(watermarked)
    
    # Apply transformations
    transformed = pipeline(watermarked)
    
    # Check watermark survival
    detected_msg = detector.detect(transformed)
    accuracy = sum(o == d for o, d in zip(original_msg, detected_msg)) / len(original_msg)
    
    print(f"Watermark survival rate: {accuracy:.2%}")
    assert accuracy > 0.8, "Watermark damaged beyond acceptable threshold"
```

## Performance Tips

### Batching

```python
def batch_transform(images: List[Image.Image], transform: Callable, batch_size: int = 8) -> List[Image.Image]:
    """Apply transform to multiple images efficiently."""
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        
        # Process batch in parallel if possible
        if hasattr(transform, 'batch_apply'):
            batch_results = transform.batch_apply(batch)
        else:
            # Fall back to sequential
            batch_results = [transform(img) for img in batch]
        
        results.extend(batch_results)
    
    return results
```

### Caching

```python
from functools import lru_cache

class CachedTransform:
    """Transform with result caching."""
    
    def __init__(self, transform: Callable, cache_size: int = 128):
        self.transform = transform
        self.cache_size = cache_size
        self._cache = {}
    
    def __call__(self, image: Image.Image) -> Image.Image:
        # Create cache key from image
        key = hash(image.tobytes())
        
        if key in self._cache:
            return self._cache[key].copy()
        
        # Apply transform
        result = self.transform(image)
        
        # Update cache
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[key] = result.copy()
        return result
```

### Vectorization

```python
def vectorized_noise(images: np.ndarray, std: float = 0.05) -> np.ndarray:
    """
    Apply noise to multiple images at once.
    
    Args:
        images: Array of shape (N, H, W, C)
        std: Noise standard deviation
        
    Returns:
        Noisy images array
    """
    noise = np.random.normal(0, std * 255, images.shape)
    noisy = np.clip(images + noise, 0, 255)
    return noisy.astype(np.uint8)
```

## Common Pitfalls

### Pitfall 1: Channel Order Confusion

**Problem**: Mixing RGB and BGR formats
**Solution**:
```python
def ensure_rgb(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """Ensure image is in RGB format."""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR if numpy array
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
    return image.convert('RGB')
```

### Pitfall 2: Loss of Precision

**Problem**: Repeated int/float conversions
**Solution**:
```python
def preserve_precision_pipeline(transforms: List[Callable]) -> Callable:
    """Pipeline that maintains float precision until the end."""
    def pipeline(image: Image.Image) -> Image.Image:
        # Convert to float once
        current = np.array(image).astype(np.float32) / 255.0
        
        # Apply all transforms in float
        for transform in transforms:
            current = transform(current)
        
        # Convert back once at the end
        current = np.clip(current * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(current)
    
    return pipeline
```

### Pitfall 3: Memory Leaks

**Problem**: Accumulating intermediate results
**Solution**:
```python
import gc

class MemoryEfficientPipeline:
    """Pipeline that explicitly manages memory."""
    
    def apply(self, image: Image.Image, transforms: List[Callable]) -> Image.Image:
        current = image
        
        for i, transform in enumerate(transforms):
            # Apply transform
            next_image = transform(current)
            
            # Explicitly delete previous if not original
            if i > 0:
                del current
                gc.collect()
            
            current = next_image
        
        return current
```

### Pitfall 4: Non-Deterministic Behavior

**Problem**: Unexpected randomness in production
**Solution**:
```python
def make_deterministic(transform: Callable, seed: int = 42) -> Callable:
    """Make any transform deterministic."""
    def deterministic_transform(image):
        # Save RNG state
        state = random.getstate()
        np_state = np.random.get_state()
        
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Apply transform
        result = transform(image)
        
        # Restore RNG state
        random.setstate(state)
        np.random.set_state(np_state)
        
        return result
    
    return deterministic_transform
```