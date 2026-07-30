"""Command-line entry point for module 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agent import LightingPromptAgent
from .client import ModelClientError
from .schemas import LightingEffectValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate a scene description into validated lighting-effect JSON."
    )
    parser.add_argument("--scene", required=True, help="Chinese or English scene description")
    parser.add_argument("--width-mm", type=float, help="fixture emitting-surface width in mm")
    parser.add_argument("--height-mm", type=float, help="fixture emitting-surface height in mm")
    parser.add_argument("--space-size-m2", type=float, help="space area in square metres")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = LightingPromptAgent().generate(
            args.scene,
            hardware_width_mm=args.width_mm,
            hardware_height_mm=args.height_mm,
            space_size_m2=args.space_size_m2,
        )
    except (ValueError, ModelClientError, LightingEffectValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
