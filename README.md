# Watermark Robustness Testing Pipeline

A user-friendly framework for testing the robustness of image watermarks across multiple watermarking methods and various image transformations.

## 📋 Overview

This repository provides a comprehensive pipeline for evaluating how well watermarks survive common image manipulations such as resizing, cropping, rotation, compression, and more. It's designed to be accessible to both technical and non-technical users.

### Supported Watermarking Methods

1. **Stable Signature** - Watermarks embedded in latent diffusion models ([Paper](https://arxiv.org/abs/2303.15435))
2. **Watermark Anything** - General-purpose watermarking system
3. **TrustMark** - Robust watermarking with quality-factor tuning

## 🎯 Who Is This For?

- **Researchers** testing watermark robustness
- **Content creators** evaluating watermarking solutions
- **Developers** integrating watermarking into applications
- **Anyone** interested in understanding watermark technology

**No deep technical knowledge required!** The notebook includes clear instructions in British English and guides you through each step.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Basic familiarity with Jupyter notebooks
- At least 4GB of disk space for models and outputs

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/facebookresearch/stable_signature
   cd stable_signature
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up watermarking models**
   
   Choose the watermarking method(s) you want to test and follow the setup instructions in the corresponding directory:
   
   - [Stable Signature Setup](watermark_models/stable-signature/README.md)
   - [Watermark Anything Setup](watermark_models/watermark-anything/README.md)
   - [TrustMark Setup](watermark_models/trustmark/README.md)

### Using the Pipeline

1. **Open the notebook**
   ```bash
   jupyter notebook pipeline_mk4.ipynb
   ```

2. **Configure your settings** in Section 1:
   - Choose your watermarking method
   - Set your username (if using Azure AI)
   - Adjust processing options

3. **Run the cells sequentially**
   - Read the instructions in each section
   - Execute cells in order
   - Monitor progress bars

4. **Review results**
   - Check the CSV files in `outputs/{method}/results/`
   - View visualisations generated in the notebook
   - Analyse watermark detection rates

## 📁 Repository Structure

```
stable_signature/
├── pipeline_mk4.ipynb              # Main user-friendly pipeline notebook
├── config.py                        # Configuration system for easy setup
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
│
├── watermark_models/               # Watermarking method implementations
│   ├── stable-signature/           # Stable Signature files
│   │   ├── models/                 # Model checkpoints
│   │   ├── watermarked_images/     # Test images
│   │   └── README.md               # Method documentation
│   ├── watermark-anything/         # Watermark Anything files
│   │   ├── models/
│   │   ├── watermarked_images/
│   │   └── README.md
│   ├── trustmark/                  # TrustMark files
│   │   ├── watermarked_images/
│   │   └── README.md
│   └── README.md                   # Adding new methods guide
│
├── outputs/                        # Generated results and transformed images
│   ├── watermark_anything/
│   │   ├── transformed_images/
│   │   └── results/
│   ├── trustmark/
│   │   ├── transformed_images/
│   │   └── results/
│   └── stable_signature/
│       ├── transformed_images/
│       └── results/
│
├── src/                            # Core library code
│   ├── ldm/                        # Latent diffusion model code
│   ├── loss/                       # Perceptual loss functions
│   └── taming/                     # VQGAN code
│
├── hidden/                         # Watermark encoder/decoder training
│   ├── main.py                     # Training script
│   ├── models.py                   # Model architectures
│   └── README.md                   # Training documentation
│
└── finetune_ldm_decoder.py        # Fine-tune LDM decoder for watermarking
```

## 🔧 Configuration

The pipeline uses a flexible configuration system that adapts to your environment:

### Using the Configuration Module

```python
from config import create_config

# Create configuration for your chosen method
config = create_config(
    method="Watermark_Anything",  # or "TrustMark", "Stable_Signature"
    user_name="YourName",          # Your Azure AI username (if applicable)
    max_images=10                  # Limit for testing (None for all images)
)

# Verify your setup
config.print_configuration()
```

### Environment Auto-Detection

The pipeline automatically detects whether you're running:
- **Azure AI**: Uses Azure-specific paths
- **Local Machine**: Uses current working directory

## 📊 Understanding the Results

After running the pipeline, you'll receive:

### CSV Results File
Contains detailed metrics for each transformed image:
- `original_image`: Name of the source image
- `transformed_image`: Name of the modified image
- `transformation_type`: Type of modification applied
- `watermark_detected`: Whether the watermark was found (True/False)
- `detected_bits`: The actual watermark bits extracted
- `psnr`: Peak Signal-to-Noise Ratio (higher = better quality)
- `ssim`: Structural Similarity Index (closer to 1 = more similar)
- `mse`: Mean Squared Error (lower = more similar)

### Visualisations
The notebook generates plots showing:
1. **Detection Rate by Transformation** - Which transformations affect watermarks most
2. **PSNR Distribution** - Overall image quality after transformations
3. **SSIM Distribution** - Structural similarity distribution
4. **Detection vs Quality** - Relationship between image quality and watermark detection

### Interpreting the Metrics

- **PSNR**: Typically 20-50 dB. Higher values indicate the transformed image looks more like the original.
- **SSIM**: Range 0-1. Values above 0.9 indicate very similar images.
- **Detection Rate**: Percentage of transformed images where the watermark was successfully detected.

## 🔬 Advanced Usage

### Testing Different Transformations

In the notebook configuration section, you can customise:

```python
# Number of images to process
MAX_IMAGES_TO_PROCESS = 10  # Start small for testing

# Enable/disable specific transformations
APPLY_RESIZE = True
APPLY_CROP = True
APPLY_ROTATION = True
APPLY_BLUR = True
APPLY_COMPRESSION = True
APPLY_NOISE = True
APPLY_COLOUR_JITTER = True

# Customise transformation parameters
RESIZE_DIMENSIONS = [(256, 256), (512, 512), (1024, 1024)]
CROP_PERCENTAGES = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50]
ROTATION_ANGLES = [5, 15, 30, 45, 90, 180]
BLUR_KERNEL_SIZES = [3, 11, 21, 51]
COMPRESSION_QUALITIES = [95, 85, 75, 50, 25]
```

### Adding Your Own Watermarking Method

See [watermark_models/README.md](watermark_models/README.md) for detailed instructions on integrating a new watermarking method.

### Training Your Own Watermark Models

For Stable Signature, you can train custom watermark models:

```bash
python finetune_ldm_decoder.py \
    --num_keys 1 \
    --ldm_config path/to/config.yaml \
    --ldm_ckpt path/to/checkpoint.pth \
    --msg_decoder_path path/to/decoder.torchscript.pt \
    --train_dir path/to/training/images \
    --val_dir path/to/validation/images
```

See [hidden/README.md](hidden/README.md) for watermark encoder/decoder training.

## 💡 Tips for Best Results

1. **Start Small**: Test with 10-20 images first to ensure everything works
2. **Check Paths**: Verify all file paths before running the full pipeline
3. **Monitor Progress**: Watch the progress bars to estimate completion time
4. **Save Regularly**: The notebook saves results after each section
5. **Multiple Methods**: Run the notebook separately for each watermarking method

## 🐛 Troubleshooting

### Common Issues

**"Root directory does not exist"**
- Update the `USER_NAME` variable in Section 1 of the notebook
- Verify you've cloned the repository to the expected location

**"No images found"**
- Check that watermarked images are in the correct directory
- Verify image file extensions (.png, .jpg, .jpeg)

**"Model checkpoint not found"**
- Download the required model files (see method-specific READMEs)
- Check the checkpoint path in the configuration

**"CUDA out of memory"**
- Reduce `MAX_IMAGES_TO_PROCESS` to process fewer images at once
- Restart the notebook kernel to clear memory

**Import errors**
- Run the dependency installation cell again
- Verify all packages in `requirements.txt` are installed

### Getting Help

If you encounter issues:
1. Check the method-specific README in `watermark_models/`
2. Review the configuration section in the notebook
3. Verify your environment setup
4. Check the original repository issues for similar problems

## 📚 Original Repository Components

This repository is based on [Stable Signature](https://github.com/facebookresearch/stable_signature) and includes:

- **Stable Diffusion Integration**: Fine-tune watermarks into LDMs
- **Watermark Training Code**: Train encoder/decoder models
- **Evaluation Scripts**: Assess robustness and image quality
- **Pre-trained Models**: Download and use existing watermark models

For the original Stable Signature documentation, see the research paper and project page.

## 📄 License

The majority of this repository is licensed under CC-BY-NC. However:
- `src/ldm` and `src/taming` are licensed under the MIT license
- Check individual watermarking methods for their specific licenses

## 🙏 Acknowledgements

This pipeline builds upon excellent work from:
- [Stable Signature](https://github.com/facebookresearch/stable_signature) by Meta AI Research
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) by Stability AI
- [Perceptual Similarity](https://github.com/SteffenCzolbe/PerceptualSimilarity)
- [HiDDeN](https://github.com/ando-khachatryan/HiDDeN) watermarking framework

## 📖 Citation

If you use this pipeline in your research, please cite the Stable Signature paper:

```bibtex
@article{fernandez2023stable,
  title={The Stable Signature: Rooting Watermarks in Latent Diffusion Models},
  author={Fernandez, Pierre and Couairon, Guillaume and J{\'e}gou, Herv{\'e} and Douze, Matthijs and Furon, Teddy},
  journal={ICCV},
  year={2023}
}
```

## 🌟 Additional Resources

- [Project Website](https://pierrefdz.github.io/publications/stablesignature/)
- [Research Paper](https://arxiv.org/abs/2303.15435)
- [Blog Post](https://ai.meta.com/blog/stable-signature-watermarking-generative-ai/)
- [Interactive Demo](https://huggingface.co/spaces/imatag/stable-signature-bzh)

---

**Happy watermarking! 🎨🔐**

For questions or contributions, please open an issue on GitHub.
