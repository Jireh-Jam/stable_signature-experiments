# Stable Signature Experiments

**Start here:** `pipeline_mk4_user_friendly.ipynb`

## What’s in this repo
- `pipeline_mk4_user_friendly.ipynb`: Guided notebook to embed, transform, and detect watermarks
- `stable_signature_experiments/`: Python package with all watermarking logic
  - `watermarking_methods/`
    - `stable_signature/`: Stable Signature implementation and pipelines
    - `watermark_anything/`: Watermark Anything implementation and CLI
    - `shared/`: Cross-method utilities (I/O, transforms)
- `pyproject.toml`: Packaging, tool configs (ruff/black/mypy)
- `.editorconfig`, `.gitignore`, `LICENSE`, `Makefile`

## Install (editable) and run
```bash
python -m pip install -U pip
pip install -e .
```

Quick import test:
```bash
python - <<'PY'
import importlib
importlib.import_module("stable_signature_experiments.watermarking_methods.stable_signature.pipelines")
importlib.import_module("stable_signature_experiments.watermarking_methods.watermark_anything.pipelines")
print("imports: OK")
PY
```

## Notebook API imports
```python
from stable_signature_experiments.watermarking_methods.stable_signature import pipelines as ss
from stable_signature_experiments.watermarking_methods.watermark_anything import pipelines as wam

# Stable Signature
out, ok = ss.run_watermark("path/to/image.jpg", message="0"*48)

# Watermark Anything
results = wam.generate_images("in/", "out/", message="0"*32, max_images=10)
```

## CLIs
After `pip install -e .`, these commands are available:
- `watermark-anything` — WAM CLI
- `watermark-anything-count-images` — Count images in a folder
- `watermark-anything-extract-matching` — Extract images matching reference names

Examples:
```bash
watermark-anything embed --input img.jpg --output wm.jpg --message 1010
watermark-anything detect --input wm.jpg
watermark-anything embed-folder --input-dir raw/ --output-dir watermarked/ --message 0xDEADBEEF
```

## Folder structure
```
.
├── README.md
├── pipeline_mk4_user_friendly.ipynb
├── pyproject.toml
├── .editorconfig
├── .gitignore
├── LICENSE
├── Makefile
└── stable_signature_experiments/
    └── watermarking_methods/
        ├── __init__.py
        ├── shared/
        │   ├── __init__.py
        │   └── transforms.py
        ├── stable_signature/
        │   ├── __init__.py
        │   ├── method.py
        │   └── pipelines.py
        └── watermark_anything/
            ├── __init__.py
            ├── __main__.py
            ├── api.py
            ├── backend.py
            ├── pipelines.py
            ├── runner.py
            └── scripts/
                ├── count_images.py
                └── extract_matching_images.py
```

## Troubleshooting
- ImportError in notebook: run `pip install -e .` from the repo root
- Torch not installed: WAM falls back to a stub; install torch for real models
- Missing checkpoints: set `--checkpoint-path` or place files under `stable_signature_experiments/watermarking_methods/watermark_anything/checkpoints/`

## Contributing
- Use absolute imports rooted at `stable_signature_experiments`
- Run format/lint/type checks:
```bash
make format
make lint
make type
```
