Advanced Attacks
=================

Overview
--------
This package provides a suite of reproducible, modular image-based watermark removal and degradation attacks. The goal is to evaluate robustness of watermarking schemes and to offer a clear API/CLI for running experiments.

Architecture
------------
```
CLI (run.py) --> attack_class.AdvancedWatermarkAttacks
                         |-- diffusion_* (optional, requires diffusers)
                         |-- adversarial_* (optional, requires foolbox)
                         |-- high_frequency_attack (FFT-based)
                         |-- analysis & reporting (metrics, plots)
```

Key Modules
-----------
- attack_class.py: Implements typed public APIs for attack methods and evaluation.
- run.py: Thin orchestrator CLI that calls the attack APIs.

Supported Attacks
-----------------
| name | description | entrypoint | flags |
|------|-------------|-----------|-------|
| high_frequency | Attenuate high-frequency components via FFT masking | AdvancedWatermarkAttacks.high_frequency_attack | --attack high_frequency --param <threshold or tuple> |
| diffusion_inpainting | Stable Diffusion inpainting guided regeneration | AdvancedWatermarkAttacks.diffusion_inpainting_attack | --attack diffusion_inpainting --param <mask_ratio> |
| diffusion_regeneration | Stable Diffusion img2img regeneration | AdvancedWatermarkAttacks.diffusion_regeneration_attack | --attack diffusion_regeneration --param <strength> |
| diffusion_image_to_image | Stable Diffusion img2img regeneration | AdvancedWatermarkAttacks.diffusion_image_to_image_attack | --attack diffusion_image_to_image --param <strength> |
| adversarial_FGSM | FGSM adversarial attack using Foolbox | AdvancedWatermarkAttacks.adversarial_attack | --attack adversarial_FGSM --param <epsilon> |
| adversarial_PGD | PGD adversarial attack using Foolbox | AdvancedWatermarkAttacks.adversarial_attack | --attack adversarial_PGD --param <epsilon> |
| adversarial_DeepFool | DeepFool adversarial attack using Foolbox | AdvancedWatermarkAttacks.adversarial_attack | --attack adversarial_DeepFool --param <epsilon> |

Quickstart
----------
Install
- Python >= 3.9
- pip install -r requirements.txt

Minimal example
```bash
python -m advanced_attacks.run \
  --original path/to/original.png \
  --watermarked path/to/watermarked.png \
  --output results \
  --attack high_frequency --param 95
```

CLI Usage
---------
```bash
python -m advanced_attacks.run --help
```
Flags:
- --original: path to original image
- --watermarked: path to watermarked image
- --output: directory to save results (idempotent; created if missing)
- --attack: one of the supported attacks (see table)
- --param: attack-specific parameter (float or tuple where noted)
- --device: "cuda" or "cpu" (defaults to auto-detect)

Python API
----------
```python
from advanced_attacks import AdvancedWatermarkAttacks

tool = AdvancedWatermarkAttacks()
metrics, attacked = tool.run_single_attack(
    original_path="orig.png",
    watermarked_path="wm.png",
    attack_type="high_frequency",
    param=95,
    output_dir="results",
)
```

Configuration
-------------
No external config required. Flags and method parameters control behavior.

Data I/O
--------
- Inputs: two image files of equal content: original and watermarked (auto resized when needed)
- Outputs: images and reports under the specified --output directory

Extending
---------
1. Add a pure function method to `AdvancedWatermarkAttacks` with type hints and docstring.
2. Log at INFO for high-level steps; DEBUG for details.
3. Update `run_single_attack` to route the new attack.

Performance & Reproducibility
-----------------------------
- Prefer setting seeds via `common.seeding.seed_everything`.
- Use appropriate device (cuda/cpu). Batch where possible.

Troubleshooting
---------------
- Missing diffusers/foolbox: install the optional deps (see requirements).
- Shape mismatches: the tool will resize outputs back to original size.

Versioning & Changelog (today)
------------------------------
- Added orchestrator `advanced_attacks/run.py`.
- Typed, logged public APIs in `attack_class.py` (non-breaking).
