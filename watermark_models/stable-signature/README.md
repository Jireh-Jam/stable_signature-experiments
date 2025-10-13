# Stable Signature

Stable Signature embeds watermarks directly into latent diffusion models (LDMs) by fine-tuning the decoder. This approach creates watermarks that are deeply integrated into the image generation process.

## Overview

- **Type**: Latent diffusion model watermarking
- **Paper**: [The Stable Signature: Rooting Watermarks in Latent Diffusion Models](https://arxiv.org/abs/2303.15435)
- **Venue**: ICCV 2023

## Setup

### 1. Download Model Weights

Download the watermark extractor model:
```bash
wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt -P models/
```

### 2. Install Dependencies

The main dependencies are already included in the repository's `requirements.txt`. Method-specific requirements:
```bash
pip install omegaconf einops torch torchvision
```

### 3. Generate Watermarked Images

To create watermarked images with Stable Signature, you'll need to:
1. Fine-tune the LDM decoder (see main repository README)
2. Generate images using the modified decoder
3. Place the generated images in the `watermarked_images/` directory

## Usage in Pipeline

The pipeline will automatically:
1. Load the TorchScript model from `models/dec_48b_whit.torchscript.pt`
2. Process images from `watermarked_images/`
3. Apply transformations and test detection robustness

## Key Features

- **Robustness**: Watermarks survive various image transformations
- **Quality**: Minimal impact on generated image quality
- **Integration**: Embedded during generation, not post-processing

## Citation

```bibtex
@article{fernandez2023stable,
  title={The Stable Signature: Rooting Watermarks in Latent Diffusion Models},
  author={Fernandez, Pierre and Couairon, Guillaume and J{\'e}gou, Herv{\'e} and Douze, Matthijs and Furon, Teddy},
  journal={ICCV},
  year={2023}
}
```

## Additional Resources

- [Project Page](https://pierrefdz.github.io/publications/stablesignature/)
- [Main Repository](https://github.com/facebookresearch/stable_signature)
- [Blog Post](https://ai.meta.com/blog/stable-signature-watermarking-generative-ai/)
