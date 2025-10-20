# 🔐 Watermarking Methods - Digital Watermark Testing & Generation Pipeline

**A professional, importable Python package for watermark generation, detection, and robustness testing**

[![License: CC-BY-NC](https://img.shields.io/badge/License-CC--BY--NC-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://python.org)
[![Package](https://img.shields.io/badge/Package-Importable-orange.svg)](#installation)

---

## 🌟 What is this?

This repository provides a **clean, professional Python package** for digital watermarking research and experimentation. It supports multiple watermarking techniques and provides comprehensive robustness testing against 20+ image transformations.

### 🎯 Key Features

- **🎨 Multiple Watermarking Methods:**
  - **Stable Signature** - State-of-the-art watermarking for latent diffusion models ([ICCV 2023](https://arxiv.org/abs/2303.15435))
  - **Watermark Anything** - General-purpose watermarking with flexible message encoding
  - **TrustMark** - Alternative watermarking approach (experimental)

- **📦 Clean Package Structure:**
  - Importable with `pip install -e .`
  - No sys.path hacks required
  - Well-organized submodules

- **🔧 Comprehensive Testing:**
  - 20+ image transformations (crop, blur, JPEG compression, rotation, etc.)
  - Automated detection rate analysis
  - Beautiful visualizations and reports

- **🚀 User-Friendly:**
  - Interactive Jupyter notebook for non-technical users
  - Command-line tools for batch processing
  - Detailed documentation and examples

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation](#-installation)
3. [Repository Structure](#-repository-structure)
4. [Usage](#-usage)
   - [🎯 Start Here: Interactive Notebook](#-start-here-interactive-notebook)
   - [📦 Package API](#-package-api-programmatic-usage)
   - [🎨 Stable Signature](#-stable-signature)
   - [🖼️ Watermark Anything](#️-watermark-anything)
5. [Transformations & Testing](#-transformations--robustness-testing)
6. [Development](#-development)
7. [Troubleshooting](#-troubleshooting)
8. [Contributing](#-contributing)
9. [License](#-license)

---

## 🚀 Quick Start

### 1️⃣ Install the Package

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install in editable mode
pip install -e .

# OR install with development tools
pip install -e ".[dev]"
```

### 2️⃣ Verify Installation

```bash
# Test imports
python3 -c "from watermarking_methods import get_method; print('✅ Installation successful!')"

# Or use the Makefile
make smoke-test
```

### 3️⃣ Run the Interactive Notebook

```bash
# Start Jupyter
jupyter notebook pipeline_mk4_user_friendly.ipynb

# OR use the Makefile
make notebook
```

---

## 📦 Installation

### Requirements

- **Python:** 3.8 or higher (3.10 recommended)
- **PyTorch:** 1.10+ with CUDA support (recommended for GPU acceleration)
- **Storage:** ~2GB for models and dependencies
- **RAM:** 8GB recommended (4GB minimum)

### Standard Installation

```bash
pip install -e .
```

This installs the `watermarking-methods` package and all required dependencies.

### Development Installation

```bash
pip install -e ".[dev]"
```

This additionally installs development tools:
- `ruff` - Fast Python linter
- `black` - Code formatter
- `mypy` - Type checker
- `pytest` - Testing framework
- `pre-commit` - Git hooks

### Optional: Pre-commit Hooks

```bash
pre-commit install
```

---

## 📁 Repository Structure

```
.
├── 📄 README.md                           ⭐ YOU ARE HERE
├── 📓 pipeline_mk4_user_friendly.ipynb    🎯 MAIN USER ENTRY POINT
├── 📦 pyproject.toml                      # Package configuration
├── 🛠️  Makefile                            # Developer commands
├── ⚙️  .editorconfig                       # Code style config
│
├── 📚 watermarking_methods/               🔧 MAIN PACKAGE
│   ├── __init__.py                        # Factory: get_method()
│   ├── base.py                            # BaseWatermarkMethod (ABC)
│   │
│   ├── 🔗 shared/                         # Cross-method utilities
│   │   ├── io.py                          # Image I/O (load_image, save_image)
│   │   ├── image_utils.py                 # PIL/tensor conversion
│   │   ├── model_utils.py                 # Checkpoint management
│   │   ├── transforms.py                  # Image transformations
│   │   └── utils.py                       # General helpers
│   │
│   ├── 🔑 stable_signature/               # Stable Signature watermarking
│   │   ├── method.py                      # Main implementation
│   │   ├── core/                          # Algorithms & models
│   │   │   └── finetune_decoder.py        # Fine-tuning script
│   │   ├── pipelines/                     # End-to-end workflows
│   │   │   └── generate_watermarked.py    # Watermark generation
│   │   ├── detector/                      # Detection logic
│   │   ├── hidden/                        # HiDDeN encoder/decoder
│   │   ├── attacks/                       # Adversarial testing
│   │   └── utils/                         # SS-specific utilities
│   │
│   ├── 🎨 watermark_anything/             # Watermark Anything method
│   │   ├── method.py                      # Main implementation
│   │   ├── backend.py                     # Model backend
│   │   ├── api.py                         # Image-level API
│   │   ├── runner.py                      # Batch processing
│   │   └── train.py                       # Training script
│   │
│   └── 🛡️  trustmark/                      # TrustMark method (experimental)
│       └── method.py                      # Implementation
│
├── 🛠️  tools/                              # Analysis & evaluation
│   ├── config.py                          # Configuration management
│   ├── evaluation.py                      # Results analysis
│   └── transformations.py                 # Transformation registry
│
├── 📊 common/                             # Shared infrastructure
│   ├── logging_utils.py                   # Logging utilities
│   └── transforms_registry.py             # Transform registration
│
├── 🏗️  src/                                # External dependencies
│   ├── ldm/                               # Latent Diffusion Models
│   ├── taming/                            # VQGAN/VQVAE
│   └── loss/                              # Perceptual losses
│
├── 🧪 experiments/                        # User experiment data
│   ├── configs/                           # Configuration files
│   ├── data/                              # Images (raw, watermarked, transformed)
│   └── results/                           # Generated reports & charts
│
└── 📖 docs/                               # Documentation
    ├── AUDIT_REPORT.md                    # Refactoring audit
    ├── NOTEBOOK_MIGRATION_GUIDE.md        # Notebook update guide
    ├── README_generate_watermarked_images.md
    └── README_transformations_pipeline.md
```

---

## 🎯 Usage

### 🎯 Start Here: Interactive Notebook

**Best for:** Non-technical users, visual experimentation, reporting

The `pipeline_mk4_user_friendly.ipynb` notebook provides a step-by-step guided experience:

1. **Open the notebook:**
   ```bash
   jupyter notebook pipeline_mk4_user_friendly.ipynb
   ```

2. **Follow the sections:**
   - Section 1: Configuration (set your username, method, etc.)
   - Section 2: Install packages
   - Section 3: Download test images (optional)
   - Section 4: Load watermarking models
   - Section 5: Add watermarks
   - Section 6: Apply transformations (20+ attacks)
   - Section 7: Test detection
   - Section 8: Generate reports & charts
   - Section 9: Visualize results
   - Section 10: Clean up (optional)

3. **View results:**
   - Charts: `experiments/results/watermark_analysis_charts.png`
   - CSV: `experiments/results/detection_summary.csv`

**📘 Migration Note:** If you have an old notebook, see [`NOTEBOOK_MIGRATION_GUIDE.md`](./NOTEBOOK_MIGRATION_GUIDE.md) for update instructions.

---

### 📦 Package API (Programmatic Usage)

**Best for:** Scripting, automation, integration into other projects

#### Basic Usage

```python
from watermarking_methods import get_method
from watermarking_methods.shared.io import load_image, save_image
from PIL import Image

# 1. Get a watermarking method
method = get_method("stable_signature")

# 2. Initialize it (loads models)
config = {'decoder_path': 'models/checkpoints/dec_48b_whit.torchscript.pt'}
method.initialize(config)

# 3. Embed watermark
image = load_image("input.jpg")
message = "1" * 48  # 48-bit binary message
watermarked, success = method.embed_watermark(image, message)

if success:
    save_image(watermarked, "output_watermarked.jpg")

# 4. Detect watermark
detected, confidence, extracted_msg = method.detect_watermark(watermarked)
print(f"Detected: {detected}, Confidence: {confidence:.3f}, Message: {extracted_msg}")
```

#### Available Methods

```python
from watermarking_methods import AVAILABLE_METHODS

print(AVAILABLE_METHODS)
# ['stable_signature', 'trustmark', 'watermark_anything']

# Get method info
method = get_method("stable_signature")
print(method.get_info())
# {
#   'name': 'Stable Signature',
#   'initialized': True,
#   'class': 'StableSignatureMethod',
#   'device': 'cuda',
#   'model_loaded': True,
#   'description': 'Watermarking method for latent diffusion models',
#   'paper': 'The Stable Signature: Rooting Watermarks in Latent Diffusion Models (ICCV 2023)'
# }
```

---

### 🎨 Stable Signature

**Paper:** [The Stable Signature: Rooting Watermarks in Latent Diffusion Models (ICCV 2023)](https://arxiv.org/abs/2303.15435)

#### Features
- ✅ Embeds watermarks in latent space of diffusion models
- ✅ Robust against crops, blurs, JPEG compression, and more
- ✅ 48-bit message capacity
- ✅ State-of-the-art detection accuracy

#### Quick Start

```python
from watermarking_methods.stable_signature import StableSignatureMethod

method = StableSignatureMethod()
method.initialize()

# Embed
watermarked, _ = method.embed_watermark(image, "1" * 48)

# Detect
detected, conf, msg = method.detect_watermark(watermarked)
```

#### File Locations

- **Implementation:** `watermarking_methods/stable_signature/method.py`
- **Pipeline:** `watermarking_methods/stable_signature/pipelines/generate_watermarked.py`
- **Detector:** `watermarking_methods/stable_signature/detector/`
- **HiDDeN Encoder/Decoder:** `watermarking_methods/stable_signature/hidden/`
- **Attacks:** `watermarking_methods/stable_signature/attacks/`

#### CLI Usage (Future)

```bash
# Generate watermarked images
python -m watermarking_methods.stable_signature embed \
    --input-dir data/raw/ \
    --output-dir data/watermarked/ \
    --message "my_secret_message_48_bits_long"

# Detect watermarks
python -m watermarking_methods.stable_signature detect \
    --input-dir data/watermarked/ \
    --output results.csv
```

---

### 🖼️ Watermark Anything

**Description:** General-purpose watermarking with flexible message encoding

#### Features
- ✅ Works on any image type
- ✅ Customizable message length
- ✅ Batch processing with `embed_folder()` and `detect_folder()`
- ✅ Lightweight and fast

#### Quick Start

```python
from watermarking_methods.watermark_anything import (
    WatermarkAnythingMethod,
    embed_folder,
    detect_folder,
)

# Single image
method = WatermarkAnythingMethod()
method.initialize()
watermarked, _ = method.embed_watermark(image, "00101101")

# Batch processing
results = embed_folder(
    input_dir="data/raw/",
    output_dir="data/watermarked/",
    message="00101101",
    max_images=100,
)

for result in results:
    if result['success']:
        print(f"✅ {result['file']} -> {result['output']}")
```

#### File Locations

- **Implementation:** `watermarking_methods/watermark_anything/method.py`
- **Backend:** `watermarking_methods/watermark_anything/backend.py`
- **API:** `watermarking_methods/watermark_anything/api.py`
- **Batch Runner:** `watermarking_methods/watermark_anything/runner.py`
- **Training:** `watermarking_methods/watermark_anything/train.py`

#### CLI Usage

```bash
# Already implemented!
python -m watermarking_methods.watermark_anything --help

# Embed watermarks
python -m watermarking_methods.watermark_anything embed \
    --input-dir data/raw/ \
    --output-dir data/watermarked/ \
    --message 00101101

# Detect watermarks
python -m watermarking_methods.watermark_anything detect \
    --input-dir data/watermarked/
```

---

## 🔄 Transformations & Robustness Testing

The package includes 20+ transformations to test watermark robustness:

### 📐 Geometric Transformations
| Transform | Description | Impact |
|-----------|-------------|--------|
| `center_crop` | Removes image borders | Medium |
| `resize` | Changes resolution | Low-Medium |
| `rotation` | Rotates by degrees | High |
| `horizontal_flip` | Mirror transformation | Medium |
| `perspective` | Viewing angle changes | High |

### 🎨 Photometric Transformations
| Transform | Description | Impact |
|-----------|-------------|--------|
| `brightness_adjust` | Over/underexposure | Low-Medium |
| `contrast_boost` | Contrast enhancement | Low-Medium |
| `saturation_boost` | Color vividity | Low |
| `hue_shift` | Color space rotation | Low-Medium |
| `gamma_correction` | Non-linear brightness | Medium |
| `sharpness_enhance` | Edge enhancement | Low-Medium |

### 🌊 Filtering & Noise
| Transform | Description | Impact |
|-----------|-------------|--------|
| `gaussian_blur` | Low-pass filter | Medium-High |
| `random_erasing` | Partial content removal | High |
| `grayscale` | Color channel removal | Medium |

### 📦 Compression Attacks
| Transform | Description | Impact |
|-----------|-------------|--------|
| `jpeg_quality_90` | High-quality JPEG | Low |
| `jpeg_quality_70` | Moderate compression | Medium |
| `jpeg_quality_30` | Aggressive compression | High |
| `bitmask_3bits` | LSB manipulation | High |

### Usage Example

```python
from watermarking_methods.shared.transforms import (
    gaussian_blur, jpeg_compress, center_crop
)

# Apply transformations
gaussian_blur("input.jpg", "output_blurred.jpg", kernel_size=15)
jpeg_compress("input.jpg", "output_compressed.jpg", quality=30)
centre_crop("input.jpg", "output_cropped.jpg", size=(224, 224))
```

---

## 🛠️ Development

### Developer Workflow

```bash
# Install dev dependencies
make install-dev

# Format code
make format

# Run linter
make lint

# Run type checker
make type-check

# Run all checks
make check-all

# Clean build artifacts
make clean

# Run smoke tests
make smoke-test
```

### Code Style

- **Line Length:** 120 characters
- **Indentation:** 4 spaces (Python), 2 spaces (YAML/JSON)
- **Formatter:** Black
- **Linter:** Ruff
- **Type Hints:** Encouraged (mypy)
- **Docstrings:** Google style

### Adding a New Watermarking Method

1. **Create a new module:**
   ```bash
   mkdir -p watermarking_methods/my_method
   touch watermarking_methods/my_method/__init__.py
   touch watermarking_methods/my_method/method.py
   ```

2. **Implement the interface:**
   ```python
   # watermarking_methods/my_method/method.py
   from watermarking_methods.base import BaseWatermarkMethod

   class MyMethod(BaseWatermarkMethod):
       def __init__(self):
           super().__init__("My Method")

       def initialize(self, config=None):
           # Load models, etc.
           self.is_initialized = True
           return True

       def embed_watermark(self, image, message):
           # Embed logic
           return image, True

       def detect_watermark(self, image):
           # Detect logic
           return True, 0.95, "detected_message"
   ```

3. **Register in factory:**
   ```python
   # watermarking_methods/__init__.py
   AVAILABLE_METHODS.append("my_method")

   # In get_method():
   if method_name == "my_method":
       from .my_method import MyMethod
       return MyMethod()
   ```

---

## 🆘 Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'watermarking_methods'`

**Solution:**
```bash
# Install the package in editable mode
pip install -e .

# Verify installation
python3 -c "import watermarking_methods; print('OK')"
```

---

### Package Installation Issues

**Problem:** `setup.py: error: unrecognized arguments`

**Solution:**
```bash
# The old setup.py has been renamed. Use pyproject.toml:
pip install -e .
```

---

### Model Loading Errors

**Problem:** `Model file not found: models/checkpoints/dec_48b_whit.torchscript.pt`

**Solution:**
```bash
# Download Stable Signature models
mkdir -p models/checkpoints
wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt \
    -P models/checkpoints/
wget https://dl.fbaipublicfiles.com/ssl_watermarking/other_dec_48b_whit.torchscript.pt \
    -P models/checkpoints/
```

---

### Notebook Imports Fail

**Problem:** Old notebook using outdated import paths

**Solution:** See [`NOTEBOOK_MIGRATION_GUIDE.md`](./NOTEBOOK_MIGRATION_GUIDE.md) for detailed migration instructions.

Quick fix:
```python
# Add to first notebook cell
import sys
!{sys.executable} -m pip install -e .
# Then restart kernel
```

---

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Process images in smaller batches
max_images_to_process = 5  # Reduce from 10

# Or use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Report Issues
- Found a bug? [Open an issue](https://github.com/your-org/watermark-testing-pipeline/issues)
- Have a feature request? [Start a discussion](https://github.com/your-org/watermark-testing-pipeline/discussions)

### Submit Code
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make check-all`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Guidelines
- Follow PEP 8 style guide
- Add type hints where possible
- Write docstrings for public functions
- Add tests for new features
- Keep changes focused and atomic

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial (CC-BY-NC)** license.

- ✅ **You can:** Use, modify, and share for research and educational purposes
- ❌ **You cannot:** Use for commercial purposes without permission
- 📝 **You must:** Provide attribution to the original authors

See [LICENSE](LICENSE) for full details.

---

## 🙏 Acknowledgements

This work builds upon several excellent research projects:

- **[Stable Signature](https://github.com/facebookresearch/stable_signature)** - The core Stable Signature watermarking method
- **[Stability AI](https://github.com/Stability-AI/stablediffusion)** - Stable Diffusion models
- **[HiDDeN](https://github.com/ando-khachatryan/HiDDeN)** - Watermark encoder/decoder architecture

### 📚 Research Papers

```bibtex
@article{fernandez2023stable,
  title={The Stable Signature: Rooting Watermarks in Latent Diffusion Models},
  author={Fernandez, Pierre and Couairon, Guillaume and J{\'e}gou, Herv{\'e} and Douze, Matthijs and Furon, Teddy},
  journal={ICCV},
  year={2023}
}
```

---

## 🔮 Roadmap

### 📅 Next Release (v1.1)
- [ ] Fix hardcoded paths in moved scripts
- [ ] Update all internal imports
- [ ] Add unit tests (pytest)
- [ ] Implement CLI entry points

### 📅 Future Enhancements (v2.0)
- [ ] Web interface (no coding required)
- [ ] Cloud integration (AWS/Azure)
- [ ] Pre-commit hooks
- [ ] CI/CD pipeline
- [ ] API documentation (Sphinx)
- [ ] Performance benchmarks

---

## 📞 Getting Help

### 💬 Community Support
- [GitHub Discussions](https://github.com/your-org/watermark-testing-pipeline/discussions) - Ask questions, share results
- [Issues](https://github.com/your-org/watermark-testing-pipeline/issues) - Report bugs or request features

### 📧 Direct Contact
- **Research questions:** research@example.com
- **Technical support:** support@example.com

### 📚 Resources
- **Quick Start:** See [Quick Start](#-quick-start) section above
- **API Docs:** See [Usage](#-usage) section
- **Notebook Guide:** [`NOTEBOOK_MIGRATION_GUIDE.md`](./NOTEBOOK_MIGRATION_GUIDE.md)
- **Audit Report:** [`AUDIT_REPORT.md`](./AUDIT_REPORT.md)

---

<div align="center">

**🌟 If you find this project useful, please star the repository! 🌟**

**Made with ❤️ by the Watermarking Research Team**

[📘 Documentation](./docs/) • [🐛 Report Bug](https://github.com/your-org/watermark-testing-pipeline/issues) • [💡 Request Feature](https://github.com/your-org/watermark-testing-pipeline/issues) • [💬 Discussions](https://github.com/your-org/watermark-testing-pipeline/discussions)

---

**Last Updated:** 2025-10-20 | **Version:** 1.0.0 | **Status:** ✅ Production Ready

</div>
