# Model Architecture: HiDDeN-based Watermarking (as used in this repo)

## Overview
This repo uses a HiDDeN-style architecture for image watermarking. It consists of:
- An encoder that takes an RGB image and a binary message, and outputs a small perturbation (watermark signal) added to the image.
- An optional JND (Just Noticeable Difference) attenuation module that scales watermark energy based on human visual sensitivity.
- An augmentation stage (for training/eval) that simulates real-world transformations.
- A decoder that recovers the embedded bits from the (possibly transformed) image.

The implementation lives primarily in `detector/models.py` (encoder/decoder, `EncoderWithJND`, `EncoderDecoder`) and `detector/attenuations.py` (JND).

## Data Flow
1. Input image `imgs` (shape b×3×H×W) and message `msgs` (shape b×K) enter the encoder.
2. Encoder produces a watermark perturbation `deltas_w` (b×3×H×W).
3. Optional channel scaling and JND heatmaps modulate `deltas_w`.
4. The final watermarked image is `imgs_w = scaling_i * imgs + scaling_w * deltas_w`.
5. During robustness evaluation, augmentations (crop/resize/blur/jpeg/etc.) are applied to `imgs_w` to simulate distortions.
6. Decoder maps the (augmented) watermarked image back to bit logits `fts` (b×K), which are thresholded at 0 to recover bits.

## Key Components
- `HiddenEncoder` (`detector/models.py`)
  - CNN that fuses message bits and image features.
  - Concatenates intermediate image representation with `msgs` broadcast to H×W, then predicts watermark deltas.
  - Optional final tanh keeps perturbation bounded.

- `EncoderWithJND` (`detector/models.py`)
  - Wraps `HiddenEncoder` to apply optional channel scaling and JND attentuation to the watermark deltas before combining with the original image.

- `JND` (`detector/attenuations.py`)
  - Computes per-pixel heatmaps from luminance (luminance masking) and gradients (contrast masking).
  - Higher heatmap values mean more watermark energy can be injected without noticeable artifacts.

- `HiddenDecoder` (`detector/models.py`)
  - CNN that predicts bit logits from the (possibly transformed) image.
  - Ends with an adaptive pooling and a linear layer to output K logits.

- `EncoderDecoder` (`detector/models.py`)
  - Full training/evaluation graph that includes attenuation and augmentation.
  - Aggregates redundancy by summing features per bit if configured.

## Notation and Shapes
- Images: `imgs` in [-1, 1] or normalized space depending on pipeline; generator scripts normalize to ImageNet stats.
- Messages: `msgs` is a binary vector of length K (e.g., K=48). At inference time, recovered bits are `fts > 0`.
- Outputs: Watermarked image `imgs_w` and decoded logits `fts`.

## Determinism and Robustness
- Determinism: Set seeds via `common.seeding.seed_everything` during scripting or experiments.
- Robustness: Use augmentation pipelines (see `tools/transformations.py` shim and `common/transformations.py`) to stress-test detection under common perturbations.

## Minimal Python Example
```python
import torch
from PIL import Image
from torchvision import transforms
from detector.models import HiddenEncoder, HiddenDecoder, EncoderWithJND
from detector.attenuations import JND

K = 48
encoder = HiddenEncoder(num_blocks=4, num_bits=K, channels=64)
decoder = HiddenDecoder(num_blocks=8, num_bits=K, channels=64)

normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
unnorm = transforms.Normalize(mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225], std=[1/0.229,1/0.224,1/0.225])
img = Image.open("example.png").convert("RGB").resize((512,512))
img_t = normalize(transforms.ToTensor()(img)).unsqueeze(0)

jnd = JND(preprocess=unnorm)
enc = EncoderWithJND(encoder, jnd, scale_channels=False, scaling_i=1.0, scaling_w=1.5)

msg_bits = torch.randint(0, 2, (1, K)).bool()
msg = 2 * msg_bits.float() - 1

img_w = enc(img_t, msg)
decoded_logits = decoder(img_w)
decoded_bits = (decoded_logits > 0)
print("Bit accuracy:", (decoded_bits == msg_bits).float().mean().item())
```

## Performance Tips
- Use GPU for faster encode/decode.
- Batch images (b > 1) where possible.
- Limit I/O; keep images as tensors during encode/decode.

## Common Pitfalls
- Mismatched normalization: ensure the same normalization at encode and decode.
- Resolution: models trained/evaluated at 512×512; resize inputs accordingly.
- Over-aggressive scaling: large `scaling_w` can degrade PSNR and visual quality.
