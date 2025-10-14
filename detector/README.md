Detector
========

Purpose
-------
This package provides watermark detection utilities built around the HiDDeN decoder model, including simple CLI and Python APIs for single-image and batch decoding.

Models and Features
-------------------
| model | input type | output | metrics |
|-------|------------|--------|---------|
| HiddenDecoder | RGB image (512x512) | bitstring of length num_bits | optional PSNR via external tools |

Pipeline Overview
-----------------
preprocessing (resize 512x512, normalize) → inference (decoder) → postprocessing (threshold, bits to string)

Usage
-----
CLI examples
```bash
python -m detector.run --images path/to/watermarked.png --ckpt hidden/ckpts/hidden_replicate.pth

python -m detector.run --images path/to/dir_of_images --ckpt hidden/ckpts/hidden_replicate.pth \
  --output-csv detector_out.csv
```

Python API
```python
from detector.watermark_detector import detect_watermark

decoded = detect_watermark("wm.png", "hidden/ckpts/hidden_replicate.pth")
print(decoded)
```

Configuration
-------------
- Threshold: fixed at 0 (>0 is bit=1). Adjust in `watermark_detector.py` if needed.
- Device: auto-detected in module; modify for explicit device selection if required.
- Batching: current CLI loops over files; extend as needed.

Evaluation
----------
Use `tools/evaluation.py` to aggregate detection performance by transformation.

Deployment Notes
----------------
- Save/load decoder weights with `torch.load` / `load_state_dict` as used in `watermark_detector.load_decoder`.
- Export to ONNX/TorchScript is not currently set up; add if needed.

Troubleshooting & FAQs
----------------------
- Ensure checkpoint path is correct and matches decoder architecture params.
- Images must be readable and convertible to RGB; script resizes to 512x512.
