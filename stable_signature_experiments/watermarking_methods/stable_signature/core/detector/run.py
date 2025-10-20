from __future__ import annotations

import argparse
from pathlib import Path

from common.logging_utils import get_logger
from .watermark_detector import detect_watermark, process_images_in_folder

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run watermark detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    single = sub.add_parser("single", help="Detect watermark in a single image")
    single.add_argument("--image", required=True, help="Path to image")
    single.add_argument("--ckpt", required=True, help="Path to hidden checkpoint")
    single.add_argument("--show", action="store_true", help="Show image briefly")

    batch = sub.add_parser("batch", help="Detect watermark across a folder")
    batch.add_argument("--folder", required=True, help="Path to folder of images")
    batch.add_argument("--ckpt", required=True, help="Path to hidden checkpoint")
    batch.add_argument("--out", default="metrics.csv", help="Output CSV path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "single":
        decoded = detect_watermark(args.image, args.ckpt, show=args.show)
        logger.info(f"Decoded message: {decoded}")
    elif args.cmd == "batch":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        process_images_in_folder(args.folder, args.ckpt, args.out)


if __name__ == "__main__":
    main()
