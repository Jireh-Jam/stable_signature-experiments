## generate_watermarked_images.py

This script embeds a 48‑bit watermark into images using a HiDDeN‑style encoder with JND attenuation, saving originals, watermarked outputs, difference visualizations, and per‑image metrics.

### Inputs/outputs
- **Input**: `--input-dir` containing subfolders of images (`*.png`, `*.jpg`).
- **Output**: `--output-dir` with subfolders: `original/`, `watermarked/`, `difference/`, `combined/`, `metrics/` and a `summary_metrics.csv`.

### CLI flags

| flag | type | default | description |
|---|---|---|---|
| `--input-dir` | str | `input` | Root directory with subfolders of images |
| `--output-dir` | str | `output` | Directory to write results |
| `--num-images` | int | None | If set, randomly processes only N images |
| `--random-msg` | bool | False | If set, use random 48‑bit message per image |

### Runnable example

```bash
python generate_watermarked_images.py \
  --input-dir hidden/imgs \
  --output-dir /tmp/wm_out \
  --num-images 5
```

### Purpose and watermark technique
- Encodes a binary message using `HiddenEncoder` and adds perceptual `JND` attenuation, then clamps and saves as RGB.
- Decoder accuracy is measured by re‑decoding the produced image and computing bit accuracy.

### Notes on quality and performance
- Resizes all inputs to 512x512 (BICUBIC) for model compatibility.
- GPU is auto‑used if available; large batches are not used here to remain simple and deterministic.

### Failure modes & recovery
- Missing checkpoint: ensure `hidden/ckpts/hidden_replicate.pth` exists; path is read relative to CWD.
- Mode mismatches: script converts to RGB; ensure non‑RGB inputs are convertible.
- Permission errors: ensure `--output-dir` is writable.

### Integration points
- Outputs can be fed into `advanced_attacks` via `--watermarked` input to evaluate robustness.
- Decoding can be validated via `detector.run single` on the produced watermarked images.
