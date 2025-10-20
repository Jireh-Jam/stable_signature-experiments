## Advanced Attacks

High-level toolkit for attacking watermarked images using diffusion-based regeneration/inpainting, adversarial perturbations, and frequency-domain operations. The goal is to evaluate watermark robustness and generate counter-examples systematically.

### Architecture

```
+-----------------------+        +------------------------+
| run.py (CLI)          |  --->  | attack_class.py        |
| - args parsing        |        | - AdvancedWatermark... |
| - orchestrates flows  |        | - attack methods       |
+-----------------------+        +------------------------+
                                     |    ^
                                     v    |
                             +--------------------+
                             | res_pipe.py (ReSD) |
                             +--------------------+
```

### Key modules & responsibilities
- `run.py`: Single entrypoint CLI to run attacks.
- `attack_class.py`: Implements `AdvancedWatermarkAttacks` with diffusion, adversarial, and frequency attacks.
- `res_pipe.py`: Custom ReSD pipeline for diffusion head-start use cases.
- `integrated_watermark_attackers.py` (optional): Extended/alternative attack variants.

### Supported attacks

| name | description | entrypoint | flags |
|------|-------------|------------|-------|
| high_frequency | reduce high-frequency components via FFT masking | `AdvancedWatermarkAttacks.high_frequency_attack` | `--param` threshold or `(threshold,strength)` via API |
| diffusion_inpainting | Stable Diffusion inpainting with center mask | `diffusion_inpainting_attack` | `--param` mask_ratio |
| diffusion_regeneration | Stable Diffusion img2img regeneration | `diffusion_regeneration_attack` | `--param` strength |
| diffusion_image_to_image | Stable Diffusion img2img | `diffusion_image_to_image_attack` | `--param` strength |
| adversarial_FGSM/PGD/DeepFool | Foolbox-based adversarial example on ImageNet model | `adversarial_attack` | `--param` epsilon |

### Quickstart

#### Install

```bash
pip install -r requirements.txt
# Optional extras
pip install diffusers[torch] transformers accelerate foolbox torchvision
```

#### Minimal runnable example

```bash
python -m advanced_attacks.run \
  --original hidden/imgs/00.png \
  --watermarked hidden/imgs/00.png \
  --output /tmp/attack_results \
  --attack high_frequency --param 95
```

### CLI

```bash
python -m advanced_attacks.run --help
```

Flags:
- `--original`: path to original image
- `--watermarked`: path to watermarked image
- `--output`: directory to save results
- `--attack`: one of the supported attacks or `all`
- `--param`: numeric parameter (meaning depends on attack)
- `--device`: `cuda` or `cpu` (auto-detected by default)

Examples:

```bash
# Run all attacks and generate a summary report
python -m advanced_attacks.run --original A.png --watermarked B.png --output out --attack all

# Diffusion inpainting with 40% mask
python -m advanced_attacks.run --original A.png --watermarked B.png --attack diffusion_inpainting --param 0.4
```

### Python API

```python
from advanced_attacks import AdvancedWatermarkAttacks

tool = AdvancedWatermarkAttacks(device='cuda')
metrics, attacked = tool.run_single_attack('A.png','B.png','high_frequency', param=(95,0.8), output_dir='out')
```

### Data I/O
- Inputs: Two image paths (`--original` and `--watermarked`) in formats readable by OpenCV/PIL.
- Outputs: Images and comparison plots saved under `--output`; a text summary if running `--attack all`.

### Extending
1. Add a pure function on `AdvancedWatermarkAttacks` that takes/returns `np.ndarray` BGR images.
2. Add it to `run_single_attack` switch.
3. Document the flag semantics in this README.

### Performance & reproducibility
- Set `PYTHONHASHSEED` and torch/np random seeds for determinism; diffusion pipelines still may vary slightly.
- Prefer `cuda` for diffusion attacks; they are slow on CPU.

### Troubleshooting
- Missing `diffusers` or `foolbox`: install the extras shown above.
- Tensor shape/device errors: ensure `--device` matches your environment; reinstall CUDA torch if needed.

### Versioning & changelog
- Consolidated orchestrator `run.py` added.
- Logging and transforms registry introduced via `common/`.
