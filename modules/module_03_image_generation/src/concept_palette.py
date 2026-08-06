"""Extract an ordered palette from a concept image and render a light field."""

from __future__ import annotations

import colorsys
from dataclasses import asdict
import re

import numpy as np
from PIL import Image

from modules.color_vocabulary import COLOR_RGB, matched_color_spans

from .structured_gradient import (
    GradientStop,
    StructuredGradientPlan,
    apply_scene_brightness_floor,
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
    # Prefer illuminated surfaces over dark furniture, beams, and window frames.
    # Dark-scene fallback below still keeps a usable palette when no bright region exists.
    mask = (maximum >= 0.50) & (saturation >= 0.08)
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
    target_hsv = colorsys.rgb_to_hsv(*(value / 255.0 for value in rgb))

    def distance(item: tuple[str, tuple[int, int, int]]) -> float:
        candidate_hsv = colorsys.rgb_to_hsv(
            *(value / 255.0 for value in item[1])
        )
        hue_delta = abs(target_hsv[0] - candidate_hsv[0])
        hue_delta = min(hue_delta, 1.0 - hue_delta)
        hue_importance = max(target_hsv[1], candidate_hsv[1])
        saturation_delta = target_hsv[1] - candidate_hsv[1]
        value_delta = target_hsv[2] - candidate_hsv[2]
        return (
            (4.0 * hue_importance * hue_delta) ** 2
            + (1.3 * saturation_delta) ** 2
            + (0.8 * value_delta) ** 2
        )

    return min(
        COLOR_RGB,
        key=distance,
    )[0]


def _merge_similar_adjacent_colors(
    colors: tuple[tuple[int, int, int], ...],
    *,
    threshold: float = 30.0,
) -> tuple[tuple[int, int, int], ...]:
    """Merge neighboring scene colors that would create redundant gradient stops."""

    merged: list[tuple[int, int, int]] = []
    for color in colors:
        if not merged:
            merged.append(color)
            continue
        distance = float(
            np.linalg.norm(
                np.asarray(color, dtype=np.float32)
                - np.asarray(merged[-1], dtype=np.float32)
            )
        )
        if distance < threshold:
            merged[-1] = tuple(
                int(round((left + right) / 2.0))
                for left, right in zip(merged[-1], color)
            )
        else:
            merged.append(color)
    return tuple(merged)


def _palette_coverage(
    concept_image: Image.Image,
    colors: tuple[tuple[int, int, int], ...],
) -> tuple[float, ...]:
    """Estimate how much of the concept image is closest to each selected color."""

    sample = np.asarray(
        concept_image.convert("RGB").resize((128, 72), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).reshape(-1, 3)
    palette = np.asarray(colors, dtype=np.float32)
    distances = np.sum((sample[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    assignments = np.argmin(distances, axis=1)
    counts = np.bincount(assignments, minlength=len(colors)).astype(np.float64)
    weights = counts / max(float(counts.sum()), 1.0)
    return tuple(float(value) for value in weights)


def build_extracted_concept_palette_plan(
    concept_image: Image.Image,
    *,
    color_count: int = 3,
) -> tuple[StructuredGradientPlan, dict[str, object]]:
    """Make the original concept image the sole source of the light-field palette."""

    extracted = extract_ordered_palette(concept_image, color_count=color_count)
    colors = _merge_similar_adjacent_colors(extracted)
    coverage = _palette_coverage(concept_image, colors)
    names = tuple(_nearest_color_name(color) for color in colors)
    positions = np.linspace(0.0, 1.0, len(colors))
    dominant_index = int(np.argmax(np.asarray(coverage, dtype=np.float32)))
    stops = tuple(
        GradientStop(
            position=float(position),
            rgb=color,
            color_name=name,
            role="primary" if index == dominant_index else "secondary",
        )
        for index, (position, color, name) in enumerate(
            zip(positions, colors, names)
        )
    )
    plan = StructuredGradientPlan(
        direction="horizontal",
        stops=stops,
        dominant_color=stops[dominant_index].color_name,
        source="concept_image_extracted_palette",
    )
    report: dict[str, object] = {
        "palette_source": "original_concept_image_only",
        "extracted_colors": [list(color) for color in extracted],
        "merged_colors": [list(color) for color in colors],
        "coverage": list(coverage),
        "requested_prompt_colors": [],
        "requested_color_weight": 0.0,
        "final_colors": [list(color) for color in colors],
        "plan": plan.to_dict(),
    }
    return plan, report


def compile_light_effect_prompt(plan: StructuredGradientPlan) -> str:
    """Describe an extracted RGB plan without making a new color decision."""

    stops = plan.stops
    if len(stops) == 1:
        placement = f"dominant {stops[0].color_name} across the entire panel"
    else:
        labels = {
            2: ("on the left", "across the center and right"),
            3: ("on the left", "through the center", "on the right"),
            4: (
                "on the far left",
                "near the left center",
                "near the right center",
                "on the far right",
            ),
        }[len(stops)]
        parts = []
        for stop, label in zip(stops, labels):
            prefix = "dominant " if stop.role == "primary" else ""
            parts.append(f"{prefix}{stop.color_name} {label}")
        placement = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return (
        f"Wide panoramic organizer-style color field with {placement}, derived from the "
        "concept scene palette, forming a clean smooth horizontal gradient with uniform "
        "vertical color, balanced illumination, natural color relationships, and an "
        "uninterrupted luminous surface throughout."
    )


def build_concept_palette_plan(
    concept_image: Image.Image,
    prompt: str,
) -> tuple[StructuredGradientPlan, dict[str, object]]:
    """Build the shared color blueprint used by both concept and light images."""

    matches = matched_color_spans(prompt)
    modified_families = {
        name.rsplit(" ", 1)[-1]
        for _start, _end, name, _rgb in matches
        if " " in name
    }
    prompt_colors: list[tuple[str, tuple[int, int, int]]] = []
    lowered = prompt.casefold()
    for start, end, name, rgb in matches:
        family = name.rsplit(" ", 1)[-1]
        context = lowered[max(0, start - 16) : min(len(lowered), end + 28)]
        has_spatial_role = bool(
            re.search(
                r"\b(?:left|right|center|centre|across|entire|dominant|dominated|"
                r"primary|main)\b",
                context,
            )
        )
        if name == family and family in modified_families and not has_spatial_role:
            continue
        if name not in {item[0] for item in prompt_colors}:
            prompt_colors.append((name, rgb))
    extraction_count = min(4, max(2, len(prompt_colors) or 4))
    extracted = extract_ordered_palette(concept_image, color_count=extraction_count)

    if prompt_colors:
        # The model-selected prompt palette is the complete color decision. Do not
        # fill unused slots with concept-image colors or invent a companion hue.
        final_colors = [
            tuple(int(value) for value in rgb) for _name, rgb in prompt_colors
        ]
        final_names = [name for name, _rgb in prompt_colors]
    else:
        final_colors = list(extracted)
        final_names = [_nearest_color_name(color) for color in extracted]

    color_count = len(final_colors)
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
    scene: str = "",
) -> tuple[Image.Image, dict[str, object]]:
    """Render a light field from the selected fast-path color blueprint."""

    base_image = render_base_gradient(plan, width=width, height=height)
    image, brightness_report = apply_scene_brightness_floor(base_image, scene)
    report = dict(palette_report)
    report.update(
        {
            "applied": True,
            "render_mode": (
                "independent_effect_fast_gradient"
                if palette_report.get("independent_chains")
                else "shared_palette_fast_gradient"
            ),
            "quality_status": "accepted",
            "brightness_floor": brightness_report,
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
