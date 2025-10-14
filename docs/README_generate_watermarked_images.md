# generate_watermarked_images.py

## Purpose
Generate watermarked images using the HiDDeN encoder with optional JND attenuation, save visualizations, and record metrics (PSNR, bit accuracy).

## Inputs/Outputs
- Inputs: images under `--input-dir`, organized in subfolders; HiDDeN checkpoint at `hidden/ckpts/hidden_replicate.pth`
- Outputs: under `--output-dir` (default `output/`):
  - `original/`, `watermarked/`, `difference/`, `combined/`, `metrics/`
  - Per-image metrics CSVs and an aggregate `summary_metrics.csv`

## CLI
| flag | type | default | description |
|------|------|---------|-------------|
| --input-dir | str | input | Base directory with images (subfolders expected) |
| --output-dir | str | output | Output root directory |
| --num-images | int | None | If set, randomly sample this many images |
| --random-msg | bool flag | False | Use random per-image watermark message |
| --ckpt | str | hidden/ckpts/hidden_replicate.pth | Path to HiDDeN checkpoint |

## Usage
```bash
python generate_watermarked_images.py \
  --input-dir hidden/imgs \
  --output-dir out_wm \
  --num-images 5
```

## Notes and Pitfalls
- Images are resized to 512x512 and normalized to ImageNet stats.
- The checkpoint path can be set via `--ckpt`; default points to `hidden/ckpts/hidden_replicate.pth`.
- If fonts/assets are missing (not used directly here), ensure PIL can load images and that directories exist (the script creates them idempotently).
- Device is auto-detected; CPU works but is slower.

## Integration
- Use outputs from `watermarked/` with `advanced_attacks` to test robustness.
- Use `detector` to decode messages from generated images and evaluate detection performance.
