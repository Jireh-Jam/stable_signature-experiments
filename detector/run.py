from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image

from .watermark_detector import detect_watermark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run watermark detector on images")
    parser.add_argument('--images', required=True, help='Path to image file or directory')
    parser.add_argument('--ckpt', required=True, help='Path to decoder checkpoint .pth')
    parser.add_argument('--output-csv', default='detector_metrics.csv', help='CSV path to write decoded messages')
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    images_path = Path(args.images)
    outputs = []

    if images_path.is_dir():
        for p in images_path.iterdir():
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                decoded = detect_watermark(str(p), args.ckpt)
                outputs.append({"image": str(p), "decoded_message": decoded})
    else:
        decoded = detect_watermark(str(images_path), args.ckpt)
        outputs.append({"image": str(images_path), "decoded_message": decoded})

    # Write CSV
    import csv
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "decoded_message"])
        writer.writeheader()
        writer.writerows(outputs)


if __name__ == "__main__":
    main()
