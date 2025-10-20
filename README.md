# 🔐 Watermarking Methods Pipeline

A comprehensive pipeline for testing and evaluating image watermarking methods, featuring **Stable Signature** and **Watermark Anything** techniques.

## 🚀 Quick Start

### 1. Install the Package

```bash
# Clone the repository
git clone https://github.com/your-org/watermarking-pipeline.git
cd watermarking-pipeline

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### 2. Run the User-Friendly Notebook

```bash
# Start Jupyter
jupyter notebook pipeline_mk4_user_friendly.ipynb
```

The notebook provides a step-by-step guide for:
- 📥 Loading images
- 🔏 Adding watermarks using different methods
- 🔄 Applying transformations to test robustness
- 📊 Evaluating watermark detection rates
- 📈 Generating visual reports

## 📁 Project Structure

```
.
├── README.md                           # You are here! 👋
├── pipeline_mk4_user_friendly.ipynb    # 🎯 START HERE - Main user notebook
├── pyproject.toml                      # Package configuration
├── Makefile                           # Convenience commands
├── .editorconfig                      # Editor settings
├── .gitignore                         # Git ignore rules
├── LICENSE                            # License information
└── stable_signature_experiments/       # Main package directory
    └── watermarking_methods/
        ├── __init__.py               # Package initialization
        ├── shared/                   # 🔧 Shared utilities
        │   ├── io.py                # Image I/O operations
        │   ├── transforms.py        # Common transformations
        │   ├── combined_transforms.py # Advanced transforms
        │   └── logging_utils.py     # Logging helpers
        ├── stable_signature/         # 🎯 Stable Signature Method
        │   ├── __init__.py
        │   ├── pipelines.py         # High-level API
        │   ├── core/                # Core algorithms
        │   ├── detector/            # Detection models
        │   ├── hidden/              # Hidden watermark models
        │   └── src/                 # Low-level implementations
        └── watermark_anything/       # 🎨 Watermark Anything Method
            ├── __init__.py
            ├── pipelines.py         # High-level API
            ├── core/                # Core models
            └── scripts/             # Utility scripts
```

## 🔬 Watermarking Methods

### 🎯 Stable Signature

A robust watermarking technique that embeds invisible signatures into images using deep learning models.

**Key Features:**
- 48-bit watermark capacity
- Robust against common image transformations
- Fast embedding and detection
- Suitable for authentication applications

**Python API:**
```python
from stable_signature_experiments.watermarking_methods.stable_signature import (
    run_watermark, embed_watermark, detect_watermark
)

# Embed watermarks in a folder
results = run_watermark(
    input_images="path/to/images",
    output_dir="path/to/output",
    message="your_48_bit_message"
)

# Embed in a single image
watermarked = embed_watermark(image, message="your_message")

# Detect watermark
detected_message, confidence = detect_watermark(watermarked_image)
```

**CLI Usage:**
```bash
# Coming soon
stable-signature embed --input images/ --output watermarked/ --message "01010..."
stable-signature detect --input watermarked/image.png
```

### 🎨 Watermark Anything

A versatile watermarking method that can embed watermarks during image generation or into existing images.

**Key Features:**
- 32-bit watermark capacity
- Works with generative models
- Batch processing support
- Multiple detection strategies

**Python API:**
```python
from stable_signature_experiments.watermarking_methods.watermark_anything import (
    generate_images, embed_folder, detect_watermark
)

# Generate watermarked images from prompts
results = generate_images(
    prompts=["A beautiful sunset", "Mountain landscape"],
    output_dir="generated/",
    watermark_message="0101..." # 32-bit
)

# Embed watermarks in existing images
results = embed_folder(
    input_dir="images/",
    output_dir="watermarked/",
    watermark_message="0101..."
)

# Detect watermark
message, confidence = detect_watermark("watermarked/image.png")
```

**CLI Usage:**
```bash
# Coming soon
watermark-anything generate --prompt "Your prompt" --output generated/
watermark-anything embed --input images/ --output watermarked/
watermark-anything detect --input watermarked/image.png
```

## 🛠️ Development

### Running Code Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all tests
make test
```

### Project Layout

- `stable_signature_experiments/watermarking_methods/` - Main package code
  - `shared/` - Utilities shared between methods
  - `stable_signature/` - Stable Signature implementation
  - `watermark_anything/` - Watermark Anything implementation
- `tests/` - Unit and integration tests
- `docs/` - Additional documentation

## 📊 Transformation Tests

The pipeline tests watermark robustness against various transformations:

### Geometric
- Center crop
- Resize
- Rotation
- Flip
- Perspective

### Photometric
- Brightness adjustment
- Contrast changes
- Saturation boost
- Hue shift
- Gamma correction

### Filtering
- Gaussian blur
- Random erasing
- Grayscale conversion

### Compression
- JPEG (multiple quality levels)
- Bit masking

## 🔍 Troubleshooting

### Import Errors

If you encounter import errors:

1. Ensure the package is installed:
   ```bash
   pip install -e .
   ```

2. Verify installation:
   ```python
   import stable_signature_experiments.watermarking_methods
   print("Package imported successfully!")
   ```

### Missing Models

Some methods require pre-trained models:

```bash
# Download Stable Signature models
wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt -P models/
wget https://dl.fbaipublicfiles.com/ssl_watermarking/other_dec_48b_whit.torchscript.pt -P models/
```

### CUDA/GPU Issues

The pipeline automatically detects GPU availability. To force CPU usage:

```python
device = 'cpu'  # Force CPU usage
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run quality checks (`make format lint type-check test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Coding Standards

- Use type hints for all function signatures
- Follow PEP 8 (enforced by Black and Ruff)
- Write docstrings for all public functions
- Add unit tests for new features
- Keep functions focused and modular

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stable Signature: Based on research from Meta AI
- Watermark Anything: Based on research from [relevant paper/team]
- Community contributors and testers

## 📧 Contact

For questions or support:
- Open an issue on GitHub
- Email: watermarking@example.com

---

**Happy Watermarking!** 🎨🔐