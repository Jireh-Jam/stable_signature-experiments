# 📓 Notebook Migration Guide

**For:** `pipeline_mk4_user_friendly.ipynb`  
**Date:** 2025-10-20  
**Purpose:** Update notebook to use new watermarking_methods package structure

---

## 🔄 Required Changes

### Cell 1: Package Installation Check

**Add this as the FIRST code cell (after markdown intro):**

```python
# ✅ Verify package installation
try:
    from watermarking_methods import get_method, AVAILABLE_METHODS
    print(f"✅ Watermarking package installed! Available methods: {', '.join(AVAILABLE_METHODS)}")
except ImportError:
    print("⚠️  Package not installed. Installing now...")
    import sys
    !{sys.executable} -m pip install -e .
    from watermarking_methods import get_method, AVAILABLE_METHODS
    print(f"✅ Installation complete! Available methods: {', '.join(AVAILABLE_METHODS)}")
```

### Cell 2: Initial Setup (UPDATE)

**BEFORE:**
```python
import sys
sys.path.append('.')
# ... other imports
```

**AFTER:**
```python
# No need for sys.path manipulation!
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import watermarking package
from watermarking_methods import get_method
from watermarking_methods.shared.io import load_image, save_image, get_image_paths

print("✅ All packages loaded successfully!")
```

### Cell 4: Load Watermarking Models (UPDATE)

**BEFORE:**
```python
if watermark_method == "Stable_Signature":
    # Manual model loading...
    pass
```

**AFTER:**
```python
print(f"🔧 Setting up {watermark_method} watermarking...")

# Use the factory to get the method
method = get_method(watermark_method)
print(f"✅ Method loaded: {method.name}")

# Initialize with config
config = {
    'decoder_path': os.path.join(models_dir, 'dec_48b_whit.torchscript.pt')
}

if method.initialize(config):
    print(f"✅ {method.name} initialized successfully!")
    print(method.get_info())
else:
    print(f"⚠️  Initialization failed. Method will operate in fallback mode.")
```

### Cell 5: Add Watermarks (UPDATE)

**BEFORE:**
```python
# Manual watermarking logic
if watermark_method == "Stable_Signature":
    # ...
elif watermark_method == "Watermark_Anything":
    # ...
```

**AFTER:**
```python
print(f"🎨 Adding watermarks using {method.name}...")

watermarked_count = 0
errors = []

for i, image_file in enumerate(image_files):
    try:
        print(f"\n🔄 Processing {i+1}/{len(image_files)}: {image_file}")
        
        # Load image using shared utilities
        image_path = os.path.join(raw_images_path, image_file)
        image = load_image(image_path)
        
        # Embed watermark
        message = "1" * 48  # Default 48-bit message
        watermarked_image, success = method.embed_watermark(image, message)
        
        if success:
            # Save using shared utilities
            output_path = os.path.join(watermarked_images_path, f"wm_{image_file}")
            save_image(watermarked_image, output_path)
            watermarked_count += 1
            print(f"✅ Success: {image_file}")
        else:
            errors.append((image_file, "Embedding failed"))
            print(f"❌ Failed: {image_file}")
            
    except Exception as e:
        errors.append((image_file, str(e)))
        print(f"❌ Error: {image_file} - {str(e)}")

print(f"\n✅ Watermarking complete: {watermarked_count}/{len(image_files)} successful")
if errors:
    print(f"⚠️  Errors: {len(errors)}")
    for file, error in errors[:5]:  # Show first 5 errors
        print(f"  - {file}: {error}")
```

### Cell 6: Apply Transformations (UPDATE)

**BEFORE:**
```python
import sys
sys.path.append('.')
from combined_transforms import resize_image, gaussian_blur, ...
```

**AFTER:**
```python
# Import transformations from the shared package
from watermarking_methods.shared.transforms import (
    # Geometric
    resize_image, centre_crop, fixed_rotation, 
    random_horizontal_flip, random_perspective,
    # Color
    color_jitter, adjust_brightness, adjust_contrast,
    adjust_saturation, adjust_hue, adjust_gamma, adjust_sharpness,
    # Filtering
    gaussian_blur, random_erasing, grayscale,
    # Compression
    jpeg_compress, bitmask_image,
    # Overlay
    overlay_text,
)

print("✅ Transformation functions loaded!")

# Define transformations dictionary (same as before)
transformations = {
    'center_crop_224': {
        'func': lambda input_path, output_path: centre_crop(input_path, output_path, size=(224, 224)),
        'description': 'Center Crop (224x224): Removes image borders',
        'impact': 'Medium'
    },
    # ... rest of transformations
}
```

### Cell 7: Test Watermark Detection (UPDATE)

**BEFORE:**
```python
def detect_watermark(image_path, method="Stable_Signature"):
    # Manual detection logic
    pass
```

**AFTER:**
```python
print("🔍 Testing watermark detection...")

detection_results = []

# Test on watermarked images
for image_file in watermarked_files:
    image_path = os.path.join(watermarked_images_path, image_file)
    image = load_image(image_path)
    
    # Detect watermark using the method
    detected, confidence, message = method.detect_watermark(image)
    
    detection_results.append({
        'image_name': image_file,
        'transformation': 'original',
        'detected': detected,
        'confidence': confidence,
        'message': message
    })
    
    status = "✅ DETECTED" if detected else "❌ NOT DETECTED"
    print(f"{status} - {image_file} (confidence: {confidence:.3f})")

# Test on transformed images
for transform_name in transformations.keys():
    print(f"\n🔍 Testing {transform_name}...")
    transform_dir = os.path.join(transformed_images_dir, transform_name)
    
    if os.path.exists(transform_dir):
        transform_files = get_image_paths(transform_dir)
        
        for image_path in transform_files:
            image = load_image(image_path)
            detected, confidence, message = method.detect_watermark(image)
            
            detection_results.append({
                'image_name': os.path.basename(image_path),
                'transformation': transform_name,
                'detected': detected,
                'confidence': confidence,
                'message': message
            })

print(f"\n✅ Detection testing complete: {len(detection_results)} tests performed")
```

---

## 🚀 Quick Migration Script

Run this in a notebook cell to auto-update imports:

```python
# Quick fix for old imports
import sys
import warnings

# Add compatibility warnings
old_imports = {
    'utils_img': 'watermarking_methods.shared.image_utils',
    'utils_model': 'watermarking_methods.shared.model_utils',
    'combined_transforms': 'watermarking_methods.shared.transforms',
}

for old, new in old_imports.items():
    if old in sys.modules:
        warnings.warn(
            f"Import '{old}' is deprecated. Use '{new}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
```

---

## 📦 New Package Structure Cheat Sheet

| Old Import | New Import |
|------------|------------|
| `from hidden.models import ...` | `from watermarking_methods.stable_signature.hidden.models import ...` |
| `from detector.watermark_detector import ...` | `from watermarking_methods.stable_signature.detector.watermark_detector import ...` |
| `import utils_img` | `from watermarking_methods.shared import image_utils` |
| `import utils_model` | `from watermarking_methods.shared import model_utils` |
| `from combined_transforms import ...` | `from watermarking_methods.shared.transforms import ...` |
| N/A (manual method selection) | `from watermarking_methods import get_method` |

---

## ✅ Testing Your Migration

After updating the notebook, run these checks:

```python
# Cell 1: Verify imports
from watermarking_methods import get_method, AVAILABLE_METHODS
from watermarking_methods.shared.io import load_image, save_image
print("✅ Imports successful!")
print(f"Available methods: {AVAILABLE_METHODS}")

# Cell 2: Test method loading
method = get_method("stable_signature")
print(f"✅ Loaded: {method.name}")
print(f"Info: {method.get_info()}")

# Cell 3: Test I/O
import tempfile
from PIL import Image

# Create a test image
test_img = Image.new('RGB', (100, 100), color='red')
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    temp_path = f.name

save_image(test_img, temp_path)
loaded_img = load_image(temp_path)
print(f"✅ I/O test passed: {loaded_img.size}")

import os
os.unlink(temp_path)
```

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'watermarking_methods'"

**Solution:**
```python
import sys
!{sys.executable} -m pip install -e .
```

### "AttributeError: module 'watermarking_methods' has no attribute 'get_method'"

**Solution:** The package installation might be stale. Restart the kernel:
```python
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

### Imports work but methods fail to initialize

**Solution:** Check that model files are in the right location:
```python
import os
models_dir = 'models/checkpoints/'
expected_files = [
    'dec_48b_whit.torchscript.pt',
    'other_dec_48b_whit.torchscript.pt',
]

for filename in expected_files:
    path = os.path.join(models_dir, filename)
    if os.path.exists(path):
        print(f"✅ Found: {filename}")
    else:
        print(f"❌ Missing: {filename}")
        print(f"   Download from: https://dl.fbaipublicfiles.com/ssl_watermarking/{filename}")
```

---

## 📚 Further Reading

- **Package Structure:** See `AUDIT_REPORT.md` for complete file reorganization
- **API Documentation:** See updated `README.md` for usage examples
- **Tooling:** Run `make smoke-test` to verify installation

---

**Last Updated:** 2025-10-20  
**Status:** Ready for migration ✅
