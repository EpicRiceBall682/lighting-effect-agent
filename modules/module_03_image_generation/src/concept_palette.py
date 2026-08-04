"""Extract an ordered palette from a concept image and render a light field."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from PIL import Image

from modules.color_vocabulary import COLOR_RGB, matched_color_spans

from .structured_gradient import (
    GradientStop,
    StructuredGradientPlan,
    render_base_gradient,
    structured_gradient_metrics,
)


def _dominant_band_color(pixels: np.ndarray) -> tuple[int, int, int]:
    flat = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    normalized = flat.astype(np.float32) / 255.0
    maximum = normalized.max(axis=1)
    minimum = normalized.min(axis=1)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 1e-6,
    )
    mask = (maximum >= 0.35) & (saturation >= 0.10)
    candidates = flat[mask]
    candidate_weights = (0.25 + saturation[mask]) * maximum[mask]
    if len(candidates) < 16:
        candidates = flat[maximum >= 0.25]
        candidate_weights = maximum[maximum >= 0.25]
    if not len(candidates):
        candidates = flat
        candidate_weights = np.ones(len(flat), dtype=np.float32)

    bins = candidates // 32
    packed = (
        bins[:, 0].astype(np.int32) * 64
        + bins[:, 1].astype(np.int32) * 8
        + bins[:, 2].astype(np.int32)
    )
    scores = np.bincount(packed, weights=candidate_weights, minlength=512)
    winner = int(np.argmax(scores))
    selected = candidates[packed == winner].astype(np.float32)
    weights = candidate_weights[packed == winner]
    color = np.average(selected, axis=0, weights=weights)
    return tuple(int(value) for value in np.rint(color))


def extract_ordered_palette(
    concept_image: Image.Image,
    *,
    color_count: int = 4,
) -> tuple[tuple[int, int, int], ...]:
    """Extract representative colors while preserving left-to-right placement."""

    if not 2 <= color_count <= 4:
        raise ValueError("color_count must be from 2 to 4")
    sample = concept_image.convert("RGB").resize((256, 144), Image.Resampling.BILINEAR)
    array = np.asarray(sample, dtype=np.uint8)
    edges = np.linspace(0, array.shape[1], color_count + 1, dtype=int)
    colors = [
        _dominant_band_color(array[:, edges[index] : edges[index + 1], :])
        for index in range(color_count)
    ]
    return tuple(colors)


def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
    target = np.asarray(rgb, dtype=np.float32)
    return min(
        COLOR_RGB,
        key=lambda item: float(
            np.sum((np.asarray(item[1], dtype=np.float32) - target) ** 2)
        ),
    )[0]


def build_concept_palette_plan(
    concept_image: Image.Image,
    prompt: str,
) -> tuple[StructuredGradientPlan, dict[str, object]]:
    """Build the shared color blueprint used by both concept and light images."""

    prompt_colors: list[tuple[str, tuple[int, int, int]]] = []
    for _start, _end, name, rgb in matched_color_spans(prompt):
        if name not in {item[0] for item in prompt_colors}:
            prompt_colors.append((name, rgb))
    color_count = min(4, max(2, len(prompt_colors) or 4))
    extracted = extract_ordered_palette(concept_image, color_count=color_count)

    final_colors = list(extracted)
    final_names = [_nearest_color_name(color) for color in extracted]
    if prompt_colors:
        prompt_positions = np.linspace(0, color_count - 1, len(prompt_colors))
        for source_index, (name, rgb) in enumerate(prompt_colors):
            target_index = int(round(float(prompt_positions[source_index])))
            # Explicit user colors are the shared contract. The stochastic concept
            # image is harmonized to this blueprint later instead of being allowed to
            # silently replace a requested hue.
            final_colors[target_index] = tuple(int(value) for value in rgb)
            final_names[target_index] = name

    positions = np.linspace(0.0, 1.0, color_count)
    stops = tuple(
        GradientStop(
            position=float(position),
            rgb=color,
            color_name=name,
            role="primary" if index >= color_count // 2 else "secondary",
        )
        for index, (position, color, name) in enumerate(
            zip(positions, final_colors, final_names)
        )
    )
    plan = StructuredGradientPlan(
        direction="horizontal",
        stops=stops,
        dominant_color=stops[-1].color_name,
        source=(
            "shared_user_palette"
            if prompt_colors
            else "concept_image_spatial_palette"
        ),
    )
    report: dict[str, object] = {
        "extracted_colors": [list(color) for color in extracted],
        "requested_prompt_colors": [
            {"name": name, "rgb": list(rgb)} for name, rgb in prompt_colors
        ],
        "requested_color_weight": 1.0 if prompt_colors else 0.0,
        "final_colors": [list(color) for color in final_colors],
        "plan": plan.to_dict(),
    }
    return plan, report


def harmonize_concept_image(
    concept_image: Image.Image,
    plan: StructuredGradientPlan,
    *,
    strength: float = 0.82,
) -> tuple[Image.Image, dict[str, object]]:
    """Match concept chroma to the shared blueprint while preserving luminance."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    source = concept_image.convert("RGB")
    target = render_base_gradient(plan, width=source.width, height=source.height)
    source_ycc = np.asarray(source.convert("YCbCr"), dtype=np.float32)
    target_ycc = np.asarray(target.convert("YCbCr"), dtype=np.float32)

    before = float(
        np.mean(np.linalg.norm(source_ycc[..., 1:] - target_ycc[..., 1:], axis=2))
    )
    harmonized_ycc = source_ycc.copy()
    harmonized_ycc[..., 1:] = (
        source_ycc[..., 1:] * (1.0 - strength)
        + target_ycc[..., 1:] * strength
    )
    after = float(
        np.mean(np.linalg.norm(harmonized_ycc[..., 1:] - target_ycc[..., 1:], axis=2))
    )
    harmonized = Image.fromarray(
        np.clip(np.rint(harmonized_ycc), 0, 255).astype(np.uint8),
        mode="YCbCr",
    ).convert("RGB")
    report: dict[str, object] = {
        "applied": strength > 0.0,
        "method": "shared_palette_ycbcr_chroma_harmonization",
        "strength": float(strength),
        "luminance_preserved": True,
        "mean_chroma_error_before": before,
        "mean_chroma_error_after": after,
        "chroma_error_reduction_fraction": (
            0.0 if before <= 1e-6 else max(0.0, 1.0 - after / before)
        ),
        "shared_palette": [asdict(stop) for stop in plan.stops],
    }
    return harmonized, report


def render_shared_palette_gradient(
    plan: StructuredGradientPlan,
    palette_report: dict[str, object],
    *,
    width: int,
    height: int,
) -> tuple[Image.Image, dict[str, object]]:
    """Render a light field from the exact blueprint used to harmonize the concept."""

    image = render_base_gradient(plan, width=width, height=height)
    report = dict(palette_report)
    report.update(
        {
            "applied": True,
            "render_mode": "shared_palette_fast_gradient",
            "quality_status": "accepted",
            "final_metrics": structured_gradient_metrics(image),
            "effective_texture_strength": 0.0,
            "post_guidance_layout_error": None,
            "post_guidance_anchor_color_error": None,
            "anchors": [asdict(stop) for stop in plan.stops],
        }
    )
    return image, report


def render_concept_palette_gradient(
    concept_image: Image.Image,
    prompt: str,
    *,
    width: int,
    height: int,
) -> tuple[Image.Image, dict[str, object]]:
    """Render the final wide light field directly from the concept palette."""

    plan, report = build_concept_palette_plan(concept_image, prompt)
    return render_shared_palette_gradient(
        plan,
        report,
        width=width,
        height=height,
    )
