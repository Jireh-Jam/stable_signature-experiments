# Transformations Pipeline

## Concept
Transformations are deterministic or stochastic image operations (e.g., crop, blur, noise, JPEG) used to test watermark robustness. The pipeline code lives in `common/transformations.py` and provides a registry of named transforms with helpers to compose them.

## Anatomy
- Registry: `ImageTransformations.get_standard_transformations()` and `.get_aggressive_transformations()` return `{name: callable}` dictionaries.
- Ordering: Use `apply_transformation_chain(image, [(name, fn), ...])` to apply a sequence.
- Parameterization: Many transforms accept factors (e.g., brightness, resize scale). The registry provides common presets; for custom params, call the underlying methods directly.
- Determinism: Set seeds via `common.seeding.seed_everything(seed)`.
- Device: Pure PIL; device-agnostic.

## Adding a New Transform
1. Implement a pure function in `ImageTransformations` with type hints and a concise docstring.
2. Optionally register a named preset in `get_standard_transformations` or `get_aggressive_transformations`.
3. Keep I/O out; operate on `PIL.Image` inputs and return a new `PIL.Image`.

Example template:
```python
from PIL import Image

@staticmethod
def my_transform(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """One-line description."""
    # ... implement ...
    return image
```

## Examples
Simple chain:
```python
from PIL import Image
from common.transformations import ImageTransformations as T

img = Image.open("wm.png").convert("RGB")
chain = [
  ("crop_10_percent", T.get_standard_transformations()["crop_10_percent"]),
  ("blur_light", T.get_standard_transformations()["blur_light"]),
]
out = T.apply_transformation_chain(img, chain)
```

Medium chain with custom params:
```python
from common.transformations import ImageTransformations as T
img = Image.open("wm.png").convert("RGB")
out = T.apply_transformation_chain(img, [
  ("resize_60", lambda im: T.resize_image(im, 0.6)),
  ("rotate_10", lambda im: T.rotate_image(im, 10.0)),
  ("jpeg_70", lambda im: T.jpeg_compression(im, 70)),
])
```

## Config-driven Usage
If using YAML/JSON configs, map names to functions via the registry and build chains accordingly. Example snippet:
```python
names = ["crop_20_percent", "noise_medium", "jpeg_50"]
reg = T.get_standard_transformations()
chain = [(n, reg[n]) for n in names]
```

## Testing New Transforms
- Quick sanity: run the transform on a small image and visually inspect.
- Ensure type/dtype is preserved and sizes are reasonable.

## Performance Tips
- Batch at a higher level (loop over images); transforms are per-image PIL ops.
- Avoid unnecessary conversions between PIL and NumPy.

## Common Pitfalls
- Shape/dtype confusion: keep PIL images in RGB.
- Over-aggressive parameters can obliterate the signal; start from presets.
