## Transformations Pipeline

Composable image transformations used to stress watermark robustness. Lives in `tools/transformations.py` with a shared registry in `common/transforms_registry.py`.

### Concept
- A transformation is a pure function `Image -> Image` (PIL).
- Pipelines are ordered chains applied sequentially.
- Determinism: seed NumPy/Python as needed for reproducibility.

### Anatomy
- `ImageTransformations`: library of standard and aggressive transforms (crop/blur/brightness/contrast/noise/jpeg, etc.).
- `registry`: a name->callable store where transforms are registered on import.
- Composition: build a list of names and resolve to callables via the registry.

### Example usage

```python
from PIL import Image
from common.transforms_registry import registry, apply_chain
from tools.transformations import ImageTransformations  # ensures registry is populated

img = Image.open("input.png").convert("RGB")
chain = registry.build_chain(["blur_light", "jpeg_70", "rotate_5"]) 
out = apply_chain(img, chain)
out.save("out.png")
```

### Config-driven usage
If you store a list of transform names in a config (YAML/JSON), read them and call `registry.build_chain(names)`.

### Adding a new transform
1. Implement a pure function `def my_transform(img: Image.Image) -> Image.Image`.
2. Register it: `registry.register("my_transform", my_transform)` (do this at import time in a module).
3. Reference it by name in pipelines.

### Testing transforms quickly
- Use a small sample image and run the example above.
- Validate output size/mode is as expected.

### Performance tips
- Prefer vectorized operations and PIL efficient filters.
- Avoid repeated conversions between PIL and NumPy when possible.

### Common pitfalls
- Shape/mode mismatches: always ensure RGB 3‑channel input when required.
- JPEG quality extremes can introduce large artifacts; document your choices.
