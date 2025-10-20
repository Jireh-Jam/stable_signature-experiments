# 🔐 Watermarking Methods - Comprehensive Testing Pipeline

**A professional toolkit for testing digital watermark robustness against image transformations**

[![License](https://img.shields.io/badge/License-CC--BY--NC-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://python.org)
[![Package](https://img.shields.io/badge/Package-Installable-orange.svg)](pyproject.toml)

---

## 🌟 What is this?

This repository provides a **production-ready, importable Python package** for testing how well digital watermarks survive common image modifications. The codebase has been **professionally refactored** with clean architecture, proper packaging, and comprehensive tooling.

### 🎯 Key Features

- **🚀 Easy Installation**: `pip install -e .` and you're ready to go
- **📦 Clean Package Structure**: Importable modules with clear separation of concerns  
- **🔧 Multiple Methods**: Stable Signature, TrustMark, and Watermark Anything
- **📊 Comprehensive Testing**: 20+ different image transformations with detailed analysis
- **📈 Professional Reports**: Automatic generation of charts, statistics, and recommendations
- **⚙️ Flexible Configuration**: YAML-based config system with CLI overrides
- **🛠️ Development Tools**: Pre-configured with ruff, black, mypy, and pytest

---

## 🚀 Quick Start Guide

### 📋 Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **4GB+ RAM** (8GB recommended for large datasets)
- **500MB+ disk space** for models and data

### 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd watermarking-methods

# Install in editable mode with all dependencies
pip install -e ".[dev,notebooks]"

# Verify installation
python -c "import watermarking_methods; print('✅ Package ready!')"
```

### 🎯 Running Your First Test

#### Option 1: Use the User-Friendly Notebook (Recommended)
```bash
# Start Jupyter and open the main notebook
jupyter notebook pipeline_mk4_user_friendly.ipynb
```

#### Option 2: Use the Command Line Interface
```bash
# Test watermark robustness on a folder of images
watermark-test --method stable_signature --input images/ --output results/

# Use a specific watermarking method
stable-signature embed-folder input_images/ watermarked_images/

# Get help
watermark-test --help
```

#### Option 3: Use the Python API
```python
from watermarking_methods import get_method
from watermarking_methods.pipelines import run_watermark_pipeline

# Initialize a watermarking method
method = get_method("stable_signature")
method.initialize()

# Run comprehensive testing pipeline
success = run_watermark_pipeline(
    method=method,
    input_dir="path/to/images",
    output_dir="path/to/results", 
    message="test_watermark"
)
```

**⏱️ Total time: 5-30 minutes depending on dataset size**

---

## 📁 Repository Structure

```
📦 watermarking-methods/
├── 📓 pipeline_mk4_user_friendly.ipynb    # 🌟 START HERE - User-friendly notebook
├── 📄 pyproject.toml                      # 📦 Package configuration & dependencies
├── 📄 Makefile                            # 🛠️ Development commands
├── 📄 .editorconfig                       # ⚙️ Code style configuration
├── 📄 README.md                           # 📖 This file
│
├── 📁 watermarking_methods/               # 🔧 Main package (importable)
│   ├── 📄 __init__.py                     # Package entry point
│   ├── 📄 base.py                         # Abstract base class for methods
│   ├── 📄 cli.py                          # Command-line interface
│   ├── 📄 pipelines.py                    # High-level testing pipelines
│   │
│   ├── 📁 shared/                         # 🔗 Common utilities
│   │   ├── 📄 __init__.py
│   │   ├── 📄 io.py                       # Image I/O functions
│   │   ├── 📄 image_utils.py              # Image processing utilities
│   │   ├── 📄 config.py                   # Configuration management
│   │   └── 📄 logging_utils.py            # Logging setup
│   │
│   ├── 📁 stable_signature/               # 🎯 Stable Signature implementation
│   │   ├── 📄 __init__.py
│   │   ├── 📄 method.py                   # Core implementation
│   │   └── 📄 cli.py                      # Method-specific CLI
│   │
│   ├── 📁 watermark_anything/             # 🌐 Watermark Anything implementation  
│   │   ├── 📄 __init__.py
│   │   ├── 📄 method.py                   # Core implementation
│   │   ├── 📄 cli.py                      # Method-specific CLI
│   │   ├── 📄 backend.py                  # Backend processing
│   │   ├── 📄 runner.py                   # Batch processing
│   │   └── 📁 scripts/                    # Utility scripts
│   │
│   └── 📁 trustmark/                      # 🛡️ TrustMark implementation
│       ├── 📄 __init__.py
│       └── 📄 method.py                   # Core implementation
│
├── 📁 tools/                              # 🛠️ Standalone utilities
│   ├── 📄 transformations.py             # Image transformation functions
│   ├── 📄 evaluation.py                  # Results analysis
│   └── 📄 config.py                      # Configuration helpers
│
├── 📁 experiments/                        # 📊 Experiment configurations
│   └── 📁 configs/
│       └── 📄 default_config.yaml        # Default settings
│
└── 📄 combined_transforms.py              # 🔄 Comprehensive transformations
```

---

## 🔧 Supported Watermarking Methods

### 🎯 Stable Signature (Recommended)
- **Description**: State-of-the-art watermarking for latent diffusion models
- **Paper**: [The Stable Signature: Rooting Watermarks in Latent Diffusion Models (ICCV 2023)](https://arxiv.org/abs/2303.15435)
- **Strengths**: Excellent robustness, research-backed, production-ready
- **CLI**: `stable-signature --help`
- **API**: `get_method("stable_signature")`

### 🌐 Watermark Anything
- **Description**: General-purpose watermarking method with broad applicability
- **Strengths**: Versatile, handles diverse image types
- **CLI**: `watermark-anything --help`  
- **API**: `get_method("watermark_anything")`

### 🛡️ TrustMark
- **Description**: Alternative watermarking approach for comparative studies
- **Strengths**: Different embedding strategy, useful for benchmarking
- **API**: `get_method("trustmark")`

---

## 🔄 Image Transformations Tested

### 📐 **Geometric Transformations**
- **Center Crop**: Tests spatial robustness by removing borders (224×224)
- **Resize**: Tests interpolation effects (512×512)
- **Rotation**: Tests geometric distortion resistance (15° rotation)
- **Horizontal Flip**: Tests mirror transformation effects
- **Perspective Transform**: Tests non-linear spatial distortions

### 🎨 **Photometric Transformations**  
- **Brightness**: Tests over/underexposed conditions (±40%)
- **Contrast**: Tests contrast enhancement (+50%)
- **Saturation**: Tests vivid color effects (+60%)
- **Hue Shift**: Tests color space rotation
- **Gamma Correction**: Tests non-linear brightness mapping (γ=1.8)
- **Sharpness**: Tests edge enhancement (2×)

### 🌊 **Filtering & Noise**
- **Gaussian Blur**: Tests low-pass filtering (σ=5, σ=15)
- **Random Erasing**: Tests partial content removal (10-20% area)
- **Grayscale**: Tests color channel removal

### 📦 **Compression**
- **JPEG**: Tests lossy encoding (Q=90, 70, 30)
- **Bit Masking**: Tests LSB manipulation (3-bit masking)

### 🔄 **Combined Attacks**
- **Color Jitter**: Multiple simultaneous photometric changes
- **Text Overlay**: Content occlusion with text

---

## 📊 Understanding Your Results

### 📈 **Detection Rates**
- **🟢 Excellent (90%+)**: Watermark survives very well
- **🟡 Good (70-89%)**: Watermark survives reasonably well  
- **🟠 Fair (50-69%)**: Watermark partially survives
- **🔴 Poor (<50%)**: Watermark struggles to survive

### 📋 **Generated Reports**
1. **📊 `summary_results.csv`**: Detection rates by transformation
2. **📄 `detailed_results.json`**: Complete results with confidence scores
3. **📈 `evaluation_report.txt`**: Human-readable summary
4. **📊 Charts**: Visual analysis (if matplotlib available)

### 🎯 **Key Metrics**
- **Overall Detection Rate**: Percentage of successful detections
- **Per-Transformation Rates**: Robustness against specific attacks
- **Confidence Scores**: Detector certainty levels
- **Vulnerability Analysis**: Which attacks are most effective

---

## ⚙️ Configuration & Customization

### 🔧 **Basic Configuration** (Notebook)
```python
# In pipeline_mk4_user_friendly.ipynb
user_name = 'Your.Username'              # Your username
watermark_method = "Stable_Signature"    # Method to test
max_images_to_process = 10               # Dataset size
```

### 📝 **Advanced Configuration** (YAML)
```yaml
# experiments/configs/custom_config.yaml
watermarking:
  method: "stable_signature"
  message_length: 48
  detection_threshold: 0.5

data:
  input_size: [256, 256]
  batch_size: 1
  max_images: 50

transformations:
  apply_standard: true
  apply_aggressive: false
  jpeg_quality_levels: [90, 70, 50, 30]
  
evaluation:
  generate_plots: true
  save_detailed_results: true
```

### 🖥️ **Command Line Options**
```bash
# Use custom config
watermark-test --config experiments/configs/custom_config.yaml

# Override specific settings
watermark-test --method watermark_anything --max-images 100

# Enable verbose logging
watermark-test --verbose --input images/ --output results/
```

---

## 🛠️ Development & Contributing

### 📦 **Development Setup**
```bash
# Install with development dependencies
pip install -e ".[dev,notebooks]"

# Run code formatting
make format

# Run linting and type checking  
make lint type

# Run tests
make test

# Start notebook server
make notebook
```

### 🧪 **Quality Gates**
- **Ruff**: Fast Python linter with auto-fixing
- **Black**: Uncompromising code formatter  
- **MyPy**: Static type checking
- **Pytest**: Comprehensive test suite

### 📝 **Code Style**
- **Line length**: 88 characters (Black standard)
- **Type hints**: Encouraged for public APIs
- **Docstrings**: Google style for all public functions
- **Import organization**: isort with ruff

### 🔄 **Development Workflow**
```bash
# Make changes to code
# ...

# Format and check code
make format lint type

# Run tests
make test

# All checks in one command
make check
```

---

## 📚 API Documentation

### 🔌 **Core API**
```python
# Import the main factory function
from watermarking_methods import get_method, AVAILABLE_METHODS

# Create a watermarking method
method = get_method("stable_signature")

# Initialize with optional config
config = {"decoder_path": "path/to/model.pt"}
success = method.initialize(config)

# Embed watermark
watermarked_image, success = method.embed_watermark(image, "my_message")

# Detect watermark  
detected, confidence, message = method.detect_watermark(image)
```

### 🔧 **Pipeline API**
```python
from watermarking_methods.pipelines import run_watermark_pipeline
from watermarking_methods.shared import load_config

# Load configuration
config = load_config("experiments/configs/default_config.yaml")

# Run complete pipeline
success = run_watermark_pipeline(
    method=method,
    input_dir="input_images/",
    output_dir="results/",
    message="test_watermark",
    config=config
)
```

### 🛠️ **Utilities API**
```python
from watermarking_methods.shared import (
    load_image, save_image,           # I/O functions
    pil_to_tensor, tensor_to_pil,     # Format conversion
    setup_logging, get_logger         # Logging
)

# Set up logging
setup_logging(verbose=True)
logger = get_logger(__name__)

# Load and process images
image = load_image("input.jpg")
tensor = pil_to_tensor(image)
# ... process tensor ...
result = tensor_to_pil(processed_tensor)
save_image(result, "output.jpg")
```

---

## 🚀 Advanced Usage

### 🔬 **Custom Experiments**
```python
# Create custom transformation pipeline
from watermarking_methods.pipelines import apply_transformations

transformations = apply_transformations(
    watermarked_images, 
    output_dir, 
    config
)

# Run detection on custom transformations
results = run_detection_evaluation(
    method, 
    watermarked_images, 
    transformed_dir, 
    results_dir, 
    config
)
```

### 📊 **Batch Processing**
```bash
# Process large datasets efficiently
watermark-test \
  --input large_dataset/ \
  --output batch_results/ \
  --method stable_signature \
  --max-images 1000 \
  --config experiments/configs/batch_config.yaml
```

### 🔧 **Custom Watermarking Methods**
```python
from watermarking_methods.base import BaseWatermarkMethod

class MyCustomMethod(BaseWatermarkMethod):
    def __init__(self):
        super().__init__("My Custom Method")
    
    def initialize(self, config=None):
        # Initialize your method
        return True
    
    def embed_watermark(self, image, message):
        # Implement watermark embedding
        return watermarked_image, success
    
    def detect_watermark(self, image):
        # Implement watermark detection
        return detected, confidence, message
```

---

## 🆘 Troubleshooting

### 🔧 **Installation Issues**
```bash
# If pip install fails, try:
pip install --upgrade pip setuptools wheel
pip install -e . --verbose

# If imports fail, check installation:
python -c "import watermarking_methods; print('OK')"
```

### 📦 **Missing Dependencies**
```bash
# Install specific dependency groups
pip install -e ".[notebooks]"  # For Jupyter support
pip install -e ".[azure]"      # For Azure storage
pip install -e ".[dev]"        # For development tools
```

### 🔍 **Import Errors**
```python
# Check package is installed correctly
import sys
print(sys.path)

# Verify package location
import watermarking_methods
print(watermarking_methods.__file__)
```

### 🐛 **Common Issues**
- **Model not found**: Download required models using setup instructions
- **CUDA errors**: Ensure PyTorch CUDA version matches your GPU drivers
- **Memory errors**: Reduce batch size or image resolution in config
- **Permission errors**: Use `--user` flag with pip install

---

## 📄 License & Citation

This project is licensed under **Creative Commons Attribution-NonCommercial (CC-BY-NC)**.

- ✅ **You can**: Use, modify, and share for research and educational purposes
- ❌ **You cannot**: Use for commercial purposes without permission  
- 📝 **You must**: Provide attribution to the original authors

### 📚 **Citation**
```bibtex
@software{watermarking_methods_2024,
  title={Watermarking Methods: A Comprehensive Testing Pipeline},
  author={Watermarking Research Team},
  year={2024},
  url={https://github.com/watermarking-research/watermarking-methods}
}
```

### 🙏 **Acknowledgements**
- **[Stable Signature](https://github.com/facebookresearch/stable_signature)** - Core watermarking method
- **[Stability AI](https://github.com/Stability-AI/stablediffusion)** - Stable Diffusion models
- **[HiDDeN](https://github.com/ando-khachatryan/HiDDeN)** - Watermark training techniques

---

## 🚀 What's Next?

### 🔮 **Upcoming Features**
- **🌐 Web Interface**: Browser-based testing (no coding required)
- **☁️ Cloud Integration**: Run experiments on cloud platforms  
- **📱 Mobile Support**: Test watermarks on mobile-processed images
- **🤖 AI Analysis**: Automatic interpretation and recommendations

### 📈 **Roadmap**
- **Q1 2024**: Enhanced CLI tools and batch processing
- **Q2 2024**: Web interface and cloud deployment
- **Q3 2024**: Mobile support and additional methods
- **Q4 2024**: AI-powered analysis and multi-language support

---

<div align="center">

**🌟 Star this repository if you find it useful! 🌟**

**Made with ❤️ by the Watermarking Research Team**

[🚀 Get Started](pipeline_mk4_user_friendly.ipynb) • [📦 Install](pyproject.toml) • [🛠️ Develop](Makefile) • [🆘 Help](#-troubleshooting)

</div>