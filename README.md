# Stable Signature Experiments

Modernised, importable repository that organises watermarking methods into a clean Python package, with a user-friendly notebook and optional CLIs.

### Start here
- Open the notebook: `pipeline_mk4_user_friendly.ipynb`
- Recommended install from repo root:
  ```bash
  python3 -m pip install -U pip
  pip install -e .[dev]
  ```

### Package layout (import path root: `stable_signature_experiments`)
```
.
├── pipeline_mk4_user_friendly.ipynb
├── pyproject.toml
├── .editorconfig
├── Makefile
└── stable_signature_experiments/
    └── watermarking_methods/
        ├── shared/                 # cross-method utilities (I/O, logging, transforms)
        ├── stable_signature/       # Stable Signature: core, pipelines, utils
        └── watermark_anything/     # Watermark Anything: API, CLI, runners
```

### Notebook imports (no sys.path hacks)
```python
from stable_signature_experiments.watermarking_methods.stable_signature import StableSignatureMethod
from stable_signature_experiments.watermarking_methods.watermark_anything import embed_folder, detect_folder
```

### CLIs
- Watermark Anything CLI:
  ```bash
  python -m stable_signature_experiments.watermarking_methods.watermark_anything embed \
    --input path/to/image.jpg --output out.jpg --message 1010
  ```

### Development
- Format: `make format` (ruff --fix, black)
- Lint: `make lint`
- Type-check: `make type`
- Smoke import: `make smoke`

### Troubleshooting
- ImportError: ensure you ran `pip install -e .[dev]` from repo root
- Missing `PIL`: `pip install Pillow`
- Missing `torch` (optional for WAM): install PyTorch per your CUDA/CPU setup

### License
See `LICENSE`.