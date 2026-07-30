"""One-command Chinese scene -> DeepSeek V4 prompt -> LoRA image pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

from modules.module_01_prompt_agent.src.agent import LightingPromptAgent
from modules.module_01_prompt_agent.src.client import ModelClientError
from modules.module_01_prompt_agent.src.schemas import LightingEffectValidationError

from .config import DEFAULT_NEGATIVE_PROMPT, GenerationConfig
from .generator import LightEffectGenerator
from .image_geometry import dimensions_from_fixture
from .model_loader import DEFAULT_BASE_MODEL, DEFAULT_LORA_PATH, ModelLoadError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a Chinese scene into an English lighting prompt and generate an image."
    )
    parser.add_argument("--scene", required=True, help="Chinese or English scene description")
    parser.add_argument("--width-mm", type=float, required=True, help="fixture width in millimetres")
    parser.add_argument("--height-mm", type=float, required=True, help="fixture height in millimetres")
    parser.add_argument("--space-size-m2", type=float, help="optional room area")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_to_image"))
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument(
        "--lora-texture-strength",
        type=float,
        default=0.10,
        help="LoRA low-frequency luminance contribution in structured mode (0-0.20)",
    )
    parser.add_argument("--legacy-diffusion", action="store_true")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser


def save_prompt_json(attributes: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = attributes.to_dict()  # LightingEffectAttributes contract
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_scene_pipeline(
    args: argparse.Namespace,
    *,
    prompt_agent: Any | None = None,
    generator_factory: Callable[..., Any] = LightEffectGenerator,
) -> tuple[object, Path, object]:
    """Run the complete module-1-to-module-3 path with injectable test doubles."""

    print("[1/2] 正在把场景转换成英文光效提示词……")
    agent = prompt_agent or LightingPromptAgent()
    attributes = agent.generate(
        args.scene,
        hardware_width_mm=args.width_mm,
        hardware_height_mm=args.height_mm,
        space_size_m2=args.space_size_m2,
    )
    prompt_path = args.output_dir / "module_01_prompt.json"
    save_prompt_json(attributes, prompt_path)
    print(f"英文提示词：{attributes.effect}")

    width, height = dimensions_from_fixture(args.width_mm, args.height_mm)
    config = GenerationConfig(
        width=width,
        height=height,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        lora_scale=args.lora_scale,
        negative_prompt=args.negative_prompt,
    )

    print("[2/2] 正在加载 LoRA 并生成 Raw 光效图……")
    generator = generator_factory(
        base_model=args.base_model,
        lora_path=args.lora,
        device=args.device,
        lora_scale=args.lora_scale,
    )
    result = generator.generate(
        attributes.effect,
        output_dir=args.output_dir,
        config=config,
        source_attributes=attributes.to_dict(),
        structured_gradient=not args.legacy_diffusion,
        lora_texture_strength=args.lora_texture_strength,
    )
    return result, prompt_path, attributes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, prompt_path, _attributes = run_scene_pipeline(args)
    except (
        ValueError,
        OSError,
        ModelClientError,
        LightingEffectValidationError,
        ModelLoadError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("完成。")
    print(f"prompt: {prompt_path.resolve()}")
    print(f"image: {result.image_path}")
    print(f"manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
