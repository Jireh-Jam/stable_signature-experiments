"""
Command-line interface for the Watermark Anything (WAM) utilities.

Examples:
  - Single image embed:
      python -m watermarking_methods.watermark_anything embed \
        --input path/to/image.jpg --output out/wm_image.jpg --message "101010..."

  - Single image detect:
      python -m watermarking_methods.watermark_anything detect \
        --input out/wm_image.jpg

  - Batch embed a folder:
      python -m watermarking_methods.watermark_anything embed-folder \
        --input-dir data/raw --output-dir data/watermarked --message "my message"

  - Batch detect in a folder:
      python -m watermarking_methods.watermark_anything detect-folder \
        --dir data/watermarked
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from . import api
from . import runner


def _build_common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-path",
        dest="checkpoint_path",
        type=str,
        default=os.environ.get("WAM_CHECKPOINT_PATH", ""),
        help="Optional path to a WAM checkpoint file (env: WAM_CHECKPOINT_PATH)",
    )
    parser.add_argument(
        "--save-json",
        dest="save_json",
        type=str,
        default=None,
        help="Optional path to save JSON results (for folder commands)",
    )


def _config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if getattr(args, "checkpoint_path", None):
        config["checkpoint_path"] = args.checkpoint_path
    return config


def cmd_embed(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    output_path, success = api.embed_on_path(
        input_path=args.input,
        output_path=args.output,
        message=args.message,
        config=config,
    )
    print(
        json.dumps(
            {
                "command": "embed",
                "input": os.path.abspath(args.input),
                "output": os.path.abspath(output_path),
                "success": bool(success),
            },
            indent=2,
        )
    )
    return 0 if success else 1


def cmd_detect(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    detected, confidence, message = api.detect_on_path(
        input_path=args.input,
        config=config,
    )
    print(
        json.dumps(
            {
                "command": "detect",
                "input": os.path.abspath(args.input),
                "detected": bool(detected),
                "confidence": float(confidence),
                "message": message,
            },
            indent=2,
        )
    )
    return 0


def cmd_embed_folder(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    results = runner.embed_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        message=args.message,
        max_images=args.max_images,
        config=config,
    )
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
    print(
        json.dumps(
            {
                "command": "embed-folder",
                "input_dir": os.path.abspath(args.input_dir),
                "output_dir": os.path.abspath(args.output_dir),
                "count": len(results),
                "ok": sum(1 for r in results if r.get("success")),
                "failed": sum(1 for r in results if not r.get("success")),
            },
            indent=2,
        )
    )
    return 0


def cmd_detect_folder(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    results = runner.detect_folder(
        dir_path=args.dir,
        config=config,
    )
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
    print(
        json.dumps(
            {
                "command": "detect-folder",
                "dir": os.path.abspath(args.dir),
                "count": len(results),
                "detected": sum(1 for r in results if r.get("detected")),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watermark_anything",
        description="CLI for Watermark Anything experiments",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_embed = sub.add_parser("embed", help="Embed a message into a single image")
    p_embed.add_argument("--input", required=True, type=str, help="Path to input image")
    p_embed.add_argument("--output", required=True, type=str, help="Path to save output image")
    p_embed.add_argument("--message", required=True, type=str, help="Message to embed")
    _build_common_parser(p_embed)
    p_embed.set_defaults(func=cmd_embed)

    p_detect = sub.add_parser("detect", help="Detect watermark in a single image")
    p_detect.add_argument("--input", required=True, type=str, help="Path to input image")
    _build_common_parser(p_detect)
    p_detect.set_defaults(func=cmd_detect)

    p_embed_folder = sub.add_parser("embed-folder", help="Embed on all images in a folder")
    p_embed_folder.add_argument("--input-dir", required=True, type=str, help="Input directory of images")
    p_embed_folder.add_argument("--output-dir", required=True, type=str, help="Directory to save outputs")
    p_embed_folder.add_argument("--message", required=True, type=str, help="Message to embed in all images")
    p_embed_folder.add_argument("--max-images", type=int, default=None, help="Optionally limit number of images")
    _build_common_parser(p_embed_folder)
    p_embed_folder.set_defaults(func=cmd_embed_folder)

    p_detect_folder = sub.add_parser("detect-folder", help="Detect on all images in a folder")
    p_detect_folder.add_argument("--dir", required=True, type=str, help="Directory of images to scan")
    _build_common_parser(p_detect_folder)
    p_detect_folder.set_defaults(func=cmd_detect_folder)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
