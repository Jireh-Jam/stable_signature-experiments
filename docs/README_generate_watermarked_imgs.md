# generate_watermarked_imgs.py

## Purpose
Legacy variant of the watermark generation script that processes images from nested folders, embeds a fixed or random message via HiDDeN, and saves outputs and metrics. Functionally similar to `generate_watermarked_images.py` but with minor implementation and default differences.

## Inputs/Outputs
- Inputs: images under `--input-dir`, organized in subfolders; expects a HiDDeN checkpoint at `ckpts/hidden_replicate.pth` (hardcoded path in this script)
- Outputs (under `--output-dir`):
  - `original/`, `watermarked/`, `difference/`, `combined/`, `metrics/`
  - Per-image metrics CSV and an aggregate `summary_metrics.csv`

## CLI
| flag | type | default | description |
|------|------|---------|-------------|
| --input-dir | str | input | Base directory with images (subfolders expected) |
| --output-dir | str | output | Output root directory |
| --num-images | int | None | If set, randomly sample this many images |
| --random-msg | bool flag | False | Use random per-image watermark message |

Note: This script uses a fixed checkpoint path (`ckpts/hidden_replicate.pth`). If your checkpoint lives elsewhere, update the code or prefer `generate_watermarked_images.py` which provides `--ckpt`.

## Usage
```bash
python generate_watermarked_imgs.py \
  --input-dir hidden/imgs \
  --output-dir out_wm_legacy \
  --num-images 5
```

## Differences vs generate_watermarked_images.py
- Hardcoded checkpoint path instead of a `--ckpt` flag.
- Minor stylistic/code differences; outputs and metrics are the same.

## Migration tip
Prefer the newer `generate_watermarked_images.py` for more flexible configuration. To keep this legacy script working without code edits, place your checkpoint at `ckpts/hidden_replicate.pth`.
