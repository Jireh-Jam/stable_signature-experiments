# Watermark Models Directory

This directory contains the different watermarking model implementations and their respective files.

## Directory Structure

```
watermark_models/
├── watermark-anything/       # Watermark Anything implementation
│   ├── models/              # Model checkpoints
│   ├── watermarked_images/  # Images watermarked with this method
│   └── README.md            # Method-specific documentation
├── trustmark/               # TrustMark implementation
│   ├── models/              # Model checkpoints (if needed)
│   ├── watermarked_images/  # Images watermarked with this method
│   └── README.md            # Method-specific documentation
└── stable-signature/        # Stable Signature implementation
    ├── models/              # Model checkpoints
    ├── watermarked_images/  # Images watermarked with this method
    └── README.md            # Method-specific documentation
```

## Adding a New Watermarking Method

To add a new watermarking method to this repository:

1. **Create a new directory** under `watermark_models/` with your method name
   ```bash
   mkdir watermark_models/your-method-name
   ```

2. **Create the required subdirectories**:
   ```bash
   cd watermark_models/your-method-name
   mkdir models watermarked_images
   ```

3. **Add your model files**:
   - Place model checkpoints in the `models/` directory
   - Add sample watermarked images to `watermarked_images/`

4. **Create a README.md** with:
   - Brief description of the method
   - Installation instructions
   - Required dependencies
   - Example usage
   - Citation information

5. **Update the main notebook** (`pipeline_mk4.ipynb`):
   - Add your method to the configuration options
   - Implement the detection function for your method
   - Add any method-specific dependencies to the installation section

## Method-Specific Documentation

For detailed information about each watermarking method, please refer to the README.md file in each method's directory.

## Model Downloads

### Stable Signature
Download the pre-trained models:
```bash
wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt -P watermark_models/stable-signature/models/
```

### Watermark Anything
Please refer to the [Watermark Anything repository](https://github.com/facebookresearch/watermark-anything) for model downloads and setup instructions.

### TrustMark
TrustMark uses built-in models from the pip package. No additional downloads required.
