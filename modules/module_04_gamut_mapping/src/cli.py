"""Command-line entry point for strict SDL gamut mapping."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

from .mapper import GamutMapper
from .sdl_palette import DEFAULT_SDL_PATH, SDLPalette


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map a raw light-effect image to exact organizer SDL colors."
    )
    parser.add_argument("--input", type=Path, required=True, help="raw image from module 3")
    parser.add_argument("--sdl", type=Path, default=DEFAULT_SDL_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/module_04"))
    parser.add_argument("--method", choices=("smooth", "nearest"), default="smooth")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dither-strength", type=float, default=1.0)
    parser.add_argument("--smooth-radius", type=float, default=0.6)
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="also save the organizer-reference nearest-color result",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"input image does not exist: {args.input}")
        palette = SDLPalette.from_file(args.sdl)
        mapper = GamutMapper(palette, batch_size=args.batch_size)
        with Image.open(args.input) as opened:
            source = opened.copy()
        result = mapper.map_image(
            source,
            method=args.method,
            dither_strength=args.dither_strength,
            smooth_radius=args.smooth_radius,
        )

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.input.stem
        image_path = output_dir / f"{stem}_sdl_{args.method}.png"
        control_path = output_dir / f"{stem}_sdl_{args.method}_control.png"
        mask_path = output_dir / f"{stem}_out_of_gamut_mask.png"
        report_path = output_dir / f"{stem}_sdl_report.json"
        result.image.save(image_path, format="PNG")
        result.control_image.save(control_path, format="PNG")
        result.out_of_gamut_mask.save(mask_path, format="PNG")

        report: dict[str, object] = {
            "input_path": str(args.input.expanduser().resolve()),
            "input_sha256": sha256_file(args.input),
            "output_path": str(image_path),
            "control_output_path": str(control_path),
            "out_of_gamut_mask_path": str(mask_path),
            "sdl_path": str(palette.source_path),
            "sdl_sha256": sha256_file(palette.source_path),
            "method": args.method,
            "configuration": {
                "batch_size": args.batch_size,
                "dither_strength": args.dither_strength,
                "smooth_radius": args.smooth_radius,
            },
            "palette": {
                "sample_count": len(palette.xy_samples),
                "unique_rgb_count": len(palette.rgb),
                "hull_xy": palette.hull_xy.tolist(),
            },
            "quality": result.quality.to_dict(),
            "quality_policy": asdict(result.quality_policy),
            "quality_status": "accepted" if result.accepted else "rejected",
            "quality_failures": list(result.quality_failures),
            "quality_advisories": list(
                result.quality_policy.advisories(result.quality)
            ),
        }
        source_manifest = args.input.with_suffix(".json")
        if source_manifest.is_file():
            report["source_manifest_path"] = str(source_manifest.resolve())
            report["source_manifest_sha256"] = sha256_file(source_manifest)

        if args.save_baseline and args.method != "nearest":
            baseline = mapper.map_image(source, method="nearest")
            baseline_path = output_dir / f"{stem}_sdl_nearest_baseline.png"
            baseline_control_path = (
                output_dir / f"{stem}_sdl_nearest_baseline_control.png"
            )
            baseline.image.save(baseline_path, format="PNG")
            baseline.control_image.save(baseline_control_path, format="PNG")
            report["baseline"] = {
                "output_path": str(baseline_path),
                "control_output_path": str(baseline_control_path),
                "quality": baseline.quality.to_dict(),
                "quality_status": (
                    "accepted" if baseline.accepted else "rejected"
                ),
                "quality_failures": list(baseline.quality_failures),
            }

        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("SDL 映射完成。")
    print(f"preview: {image_path}")
    print(f"control: {control_path}")
    print(f"mask: {mask_path}")
    print(f"report: {report_path}")
    print(
        "strict invalid pixels: "
        f"{result.quality.strict_invalid_pixel_count}/{result.quality.pixel_count}"
    )
    if result.quality_failures:
        print(
            "quality gate rejected the mapped result: "
            + "; ".join(result.quality_failures),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
