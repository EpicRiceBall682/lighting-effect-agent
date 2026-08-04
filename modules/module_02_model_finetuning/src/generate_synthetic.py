"""Create synthetic wide-format lighting images using the supplied reference methods."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from generate_sky_light_texture import LAYOUTS, generate_sky_light_texture, resolve_layout
from simulate_graph_more import generate_gradient_image as generate_fluid_image
from simulate_graph_one import generate_gradient_image as generate_linear_image
from simulate_graph_one import random_color_pair

from .image_palette import audit_image_file, audit_rgb_pixels


SKY_STYLES = ("sunset", "dawn", "lavender", "golden")


def _hsv(rgb: np.ndarray) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(*(float(channel) / 255.0 for channel in rgb))


def _relative_luminance(rgb: np.ndarray) -> float:
    srgb = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])


def palette_is_allowed(colors: list[np.ndarray] | tuple[np.ndarray, ...]) -> bool:
    """Accept every hue while keeping synthetic light fields sufficiently bright."""

    for color in colors:
        hue, saturation, value = _hsv(color)
        if value < 0.68:
            return False
        # HSV value alone labels saturated blue as "bright" even when it looks dark.
        if _relative_luminance(color) < 0.20:
            return False
    return True


def color_name(rgb: np.ndarray) -> str:
    hue, saturation, value = _hsv(rgb)
    if saturation < 0.16:
        return "soft ivory" if value > 0.86 else "warm neutral"
    if hue < 0.025 or hue >= 0.975:
        return "warm red"
    if hue < 0.055:
        return "soft coral"
    if hue < 0.095:
        return "warm orange"
    if hue < 0.155:
        return "golden amber"
    if hue < 0.19:
        return "pale yellow"
    if hue < 0.30:
        return "bright green"
    if hue < 0.52:
        return "bright cyan"
    if hue < 0.64:
        return "light blue"
    if hue < 0.76:
        return "soft lavender"
    if hue < 0.88:
        return "soft magenta"
    return "warm pink"


def _write_metadata(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def generate_synthetic_dataset(
    output_dir: Path,
    *,
    linear_count: int,
    fluid_count: int,
    sky_count: int,
    width: int,
    height: int,
    seed: int,
) -> Path:
    if min(linear_count, fluid_count, sky_count) < 0:
        raise ValueError("image counts cannot be negative")
    if width < 64 or height < 64:
        raise ValueError("width and height must be at least 64 pixels")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    linear_rng = np.random.default_rng(seed)
    produced = attempts = 0
    while produced < linear_count:
        attempts += 1
        if attempts > max(100, linear_count * 30):
            raise RuntimeError("could not generate enough allowed linear palettes")
        left, right = random_color_pair(linear_rng)
        if not palette_is_allowed([left, right]):
            continue
        image = generate_linear_image(width, height, left, right, linear_rng)
        palette_audit = audit_rgb_pixels(image)
        if not palette_audit.allowed:
            continue
        filename = f"linear_{produced + 1:04d}.png"
        Image.fromarray(image, "RGB").save(image_dir / filename, optimize=True)
        left_name = color_name(left)
        right_name = color_name(right)
        palette_family = "|".join(sorted((left_name, right_name)))
        recipe_id = f"linear:{left_name}->{right_name}"
        records.append(
            {
                "file_name": f"images/{filename}",
                "text": (
                    f"Wide panoramic lighting with a seamless horizontal gradient transitioning "
                    f"from {left_name} to {right_name}, a soft central mist, gentle "
                    "luminosity, and clean continuous color blending."
                ),
                "source": "synthetic_linear",
                "caption_source": "template",
                "palette_audit": palette_audit.to_dict(),
                "recipe_id": recipe_id,
                "split_group": f"synthetic_linear:{recipe_id}",
                "palette_family": palette_family,
                "layout_id": "horizontal",
                "seed": seed,
                "width": width,
                "height": height,
            }
        )
        produced += 1

    fluid_rng = np.random.default_rng(seed + 100_000)
    produced = attempts = 0
    while produced < fluid_count:
        attempts += 1
        if attempts > max(100, fluid_count * 30):
            raise RuntimeError("could not generate enough allowed fluid palettes")
        image, mode, colors = generate_fluid_image(width, height, fluid_rng)
        if not palette_is_allowed(colors):
            continue
        palette_audit = audit_rgb_pixels(image)
        if not palette_audit.allowed:
            continue
        filename = f"fluid_{produced + 1:04d}.png"
        Image.fromarray(image, "RGB").save(image_dir / filename, optimize=True)
        names = list(dict.fromkeys(color_name(color) for color in colors))
        palette_family = "|".join(sorted(names))
        recipe_id = f"fluid:{mode}:{palette_family}"
        records.append(
            {
                "file_name": f"images/{filename}",
                "text": (
                    f"Wide panoramic {mode} gradient lighting blending {', '.join(names)}, with "
                    "soft diffused glows, an airy bright tone, and smooth flowing transitions."
                ),
                "source": "synthetic_fluid",
                "caption_source": "template",
                "palette_audit": palette_audit.to_dict(),
                "recipe_id": recipe_id,
                "split_group": f"synthetic_fluid:{recipe_id}",
                "palette_family": palette_family,
                "layout_id": mode,
                "seed": seed + 100_000,
                "generator_mode": mode,
                "width": width,
                "height": height,
            }
        )
        produced += 1

    produced = attempts = 0
    candidate_path = image_dir / ".sky_candidate.png"
    while produced < sky_count:
        attempts += 1
        if attempts > max(100, sky_count * 30):
            raise RuntimeError("could not generate enough allowed sky palettes")
        item_seed = seed + 200_000 + attempts - 1
        style = SKY_STYLES[(attempts - 1) % len(SKY_STYLES)]
        layout = resolve_layout("auto", item_seed)
        generate_sky_light_texture(
            candidate_path,
            width=width,
            height=height,
            seed=item_seed,
            style=style,
            layout=layout,
        )
        palette_audit = audit_image_file(candidate_path)
        if not palette_audit.allowed:
            candidate_path.unlink(missing_ok=True)
            continue
        filename = f"sky_{produced + 1:04d}.png"
        candidate_path.replace(image_dir / filename)
        style_text = style.replace("blue", "bright blue sky").replace("golden", "golden daylight")
        recipe_id = f"sky:{style}:{layout}"
        records.append(
            {
                "file_name": f"images/{filename}",
                "text": (
                    f"Wide panoramic {style_text} lighting with a {layout.replace('_', ' ')} color "
                    "flow, broad atmospheric diffusion, gentle luminous haze, and seamless "
                    "low-frequency transitions."
                ),
                "source": "synthetic_sky",
                "caption_source": "template",
                "palette_audit": palette_audit.to_dict(),
                "recipe_id": recipe_id,
                "split_group": f"synthetic_sky:{recipe_id}",
                "palette_family": style,
                "layout_id": layout,
                "seed": item_seed,
                "generator_style": style,
                "generator_layout": layout,
                "width": width,
                "height": height,
            }
        )
        produced += 1

    metadata_path = output_dir / "synthetic_metadata.jsonl"
    _write_metadata(metadata_path, records)
    summary = {
        "image_count": len(records),
        "counts": {
            "synthetic_linear": linear_count,
            "synthetic_fluid": fluid_count,
            "synthetic_sky": sky_count,
        },
        "width": width,
        "height": height,
        "seed": seed,
        "metadata": metadata_path.name,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic wide-format light-effect data.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--linear-count", type=int, default=60)
    parser.add_argument("--fluid-count", type=int, default=60)
    parser.add_argument("--sky-count", type=int, default=60)
    parser.add_argument("--width", type=int, default=1056)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(argv)
    metadata = generate_synthetic_dataset(
        args.output_dir,
        linear_count=args.linear_count,
        fluid_count=args.fluid_count,
        sky_count=args.sky_count,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(f"generated metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
