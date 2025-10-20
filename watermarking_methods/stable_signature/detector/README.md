## Detector

Watermark detection utilities built around the HiDDeN-style encoder/decoder. Supports single-image and batch detection via a clean CLI.

### Purpose and scope
- **Goal**: extract embedded watermark bits from images and report metrics.
- **Scope**: inference-time decoding only; training is out of scope here.

### Models and features

| model | input | output | notes |
|---|---|---|---|
| HiddenDecoder | 512x512 RGB | bit logits (float), thresholded to bits | Loaded from checkpoint

### Pipeline overview
- preprocessing: resize to 512x512, ImageNet normalize
- inference: `HiddenDecoder(img_tensor)`
- postprocessing: threshold > 0 to bits, stringify

### Usage (CLI)

```bash
# Single image
python -m detector.run single --image path/to/img.png --ckpt hidden/ckpts/hidden_replicate.pth

# Batch folder
python -m detector.run batch --folder path/to/images --ckpt hidden/ckpts/hidden_replicate.pth --out /tmp/metrics.csv
```

### Python API

```python
from detector.watermark_detector import detect_watermark
msg = detect_watermark("img.png", "hidden/ckpts/hidden_replicate.pth", show=False)
print(msg)
```

### Configuration
- Thresholding: fixed (>0). Adjust in code if needed for research.
- Device: auto (`cuda` if available) via torch.
- Batching: current API processes one image at a time; batch helper writes CSV.

### Evaluation
- The batch CLI writes `decoded_message` per file to CSV for downstream scoring.

### Deployment notes
- Save/Load uses `torch.load(ckpt)['encoder_decoder']` and extracts decoder weights.
- For ONNX/TorchScript, export the `HiddenDecoder` with a dummy input of shape (1,3,512,512).

### Troubleshooting
- Wrong size: inputs are resized internally to 512x512.
- GPU errors: ensure `torch` with CUDA is installed; pass through `CUDA_VISIBLE_DEVICES`.
