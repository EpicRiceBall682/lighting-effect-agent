"""Command-line entry point for module 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import DEFAULT_NEGATIVE_PROMPT, GenerationConfig
from .generator import LightEffectGenerator
from .image_geometry import dimensions_from_fixture
from .model_loader import DEFAULT_BASE_MODEL, DEFAULT_LORA_PATH, ModelLoadError

from modules.module_01_prompt_agent.src.schemas import LightingEffectAttributes


def prompt_from_json(path: Path) -> str:
    return attributes_from_json(path).effect


def attributes_from_json(path: Path) -> LightingEffectAttributes:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LightingEffectAttributes.from_mapping(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a raw lighting-effect image with SD 1.5 LoRA.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="English lighting-effect prompt from module 1")
    prompt_group.add_argument("--prompt-json", type=Path, help="module 1 JSON containing an effect field")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/module_03"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fixture-width-mm", type=float)
    parser.add_argument("--fixture-height-mm", type=float)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--lora-texture-strength",
        type=float,
        default=0.10,
        help="LoRA low-frequency luminance contribution in structured mode (0-0.20)",
    )
    parser.add_argument(
        "--legacy-diffusion",
        action="store_true",
        help="use the previous diffusion-color output instead of the organizer-style gradient",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--no-enrich", action="store_true", help="do not append the standard texture suffix")
    return parser


def _dimensions(args: argparse.Namespace) -> tuple[int, int]:
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be provided together")
    if args.width is not None:
        return args.width, args.height
    if (args.fixture_width_mm is None) != (args.fixture_height_mm is None):
        raise ValueError("fixture width and height must be provided together")
    if args.fixture_width_mm is not None:
        return dimensions_from_fixture(args.fixture_width_mm, args.fixture_height_mm)
    return 1024, 320


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        attributes = None if args.prompt is not None else attributes_from_json(args.prompt_json)
        prompt = args.prompt if args.prompt is not None else attributes.effect
        width, height = _dimensions(args)
        config = GenerationConfig(
            width=width,
            height=height,
            seed=args.seed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            lora_scale=args.lora_scale,
            negative_prompt=args.negative_prompt,
        )
        generator = LightEffectGenerator(
            base_model=args.base_model,
            lora_path=args.lora,
            device=args.device,
            lora_scale=config.lora_scale,
        )
        result = generator.generate(
            prompt,
            output_dir=args.output_dir,
            config=config,
            enrich=not args.no_enrich,
            source_attributes=None if attributes is None else attributes.to_dict(),
            structured_gradient=not args.legacy_diffusion,
            lora_texture_strength=args.lora_texture_strength,
        )
    except (ValueError, OSError, ModelLoadError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"image: {result.image_path}")
    print(f"manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
