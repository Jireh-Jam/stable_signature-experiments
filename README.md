# Watermark Experiments – Easy Start

This repository contains reproducible experiments for image watermarking, including the Stable Signature method. It now includes a simple command-line tool and a step‑by‑step notebook aimed at non‑technical users.

## Quick start

- **1. Create an environment** (Python 3.8+ recommended):
  ```bash
  pip install -r requirements.txt
  ```
- **2. Download Stable Signature decoder weights** (if you plan to evaluate it):
  ```bash
  mkdir -p models
  wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt -P models/
  ```
- **3. Put your images in a folder** (e.g. `data/watermarked`).

## Run an evaluation (no coding)

Use the simple CLI to test detection or decoding across images:

```bash
python tools/wm_cli.py \
  --model stable_signature \
  --img_dir data/watermarked \
  --key_str 111010110101000001010111010011010100010000100111 \
  --attack_mode few \
  --output_dir output/cli
```

- **Results**: CSV files are written to `output/cli/`.
- To only decode bits (no key check), add `--decode_only` and omit `--key_str`.
- To compute image quality metrics against a non‑watermarked folder, add `--img_dir_nw path/to/non_watermarked`.

## Notebook for non‑technical users

Open `Pipeline_mk4.ipynb` and follow it top‑to‑bottom. Each step is written in clear British English and guides you through:
- Selecting a watermarking method
- Pointing to your image folders
- Running detection/decoding
- Exporting results to CSV for sharing

The notebook uses the same CLI under the bonnet, so it is resilient and easy to repeat. It works locally and in cloud notebooks.

## Pluggable models (advanced)

We added a small interface so new watermarking models can be plugged in without changing the notebook:
- Implement `WatermarkModel` in `watermarking/base.py`
- Register it with `@register_model("your_name")`
- Expose it by importing the module in `watermarking/models/__init__.py`

Your model will then be available via `--model your_name` in the CLI and notebook.

## Original research docs

The project is based on “The Stable Signature: Rooting Watermarks in Latent Diffusion Models.” For detailed setup, training, and background, see the original sections below or `hidden/README.md`.

---

