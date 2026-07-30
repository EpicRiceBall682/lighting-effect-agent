"""Command-line entry point for module-five pattern generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .generator import PatternGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a themed luminaire pattern.")
    parser.add_argument("--input", type=Path, required=True, help="module-three RGB image")
    parser.add_argument("--scene", required=True, help="Chinese or English scene description")
    parser.add_argument("--attributes-json", type=Path)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--disable-pattern", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/module_05"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    module_01_attributes = None
    if args.attributes_json:
        module_01_attributes = json.loads(
            args.attributes_json.read_text(encoding="utf-8")
        )
    with Image.open(args.input) as opened:
        base_image = opened.convert("RGB").copy()
    result = PatternGenerator().generate(
        base_image,
        scene=args.scene,
        module_01_attributes=module_01_attributes,
        seed=args.seed,
        output_dir=args.output_dir,
        pattern_strength=args.strength,
        enabled=not args.disable_pattern,
    )
    print(f"image: {result.image_path}")
    print(f"manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
