"""Organizer-faithful horizontal gradients with restrained LoRA light texture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import numpy as np
from PIL import Image, ImageFilter

from modules.color_vocabulary import (
    matched_color_spans,
    unsupported_color_terms,
)
from modules.linear_luminance import (
    apply_chromaticity_preserving_gain,
    blend_with_white_linear,
    relative_luminance_rgb8,
)

from .prompt_guidance import extract_color_anchors


MAX_TEXTURE_STRENGTH = 0.20
DEFAULT_TEXTURE_STRENGTH = 0.10
MAX_VERTICAL_VARIATION = 0.018


@dataclass(frozen=True, slots=True)
class BrightnessPolicy:
    mode: str
    target_mean_luminance: float
    target_p10_luminance: float
    maximum_chromaticity_gain: float
    maximum_white_mix: float
    minimum_mean_channel_peak: float
    minimum_p10_channel_peak: float


STANDARD_BRIGHTNESS_POLICY = BrightnessPolicy(
    mode="standard",
    target_mean_luminance=0.35,
    target_p10_luminance=0.18,
    maximum_chromaticity_gain=1.35,
    maximum_white_mix=0.22,
    minimum_mean_channel_peak=0.70,
    minimum_p10_channel_peak=0.48,
)
ENERGETIC_BRIGHTNESS_POLICY = BrightnessPolicy(
    mode="energetic",
    target_mean_luminance=0.32,
    target_p10_luminance=0.20,
    maximum_chromaticity_gain=1.35,
    maximum_white_mix=0.24,
    minimum_mean_channel_peak=0.74,
    minimum_p10_channel_peak=0.52,
)
DARK_BRIGHTNESS_POLICY = BrightnessPolicy(
    mode="intentional_dark",
    target_mean_luminance=0.24,
    target_p10_luminance=0.10,
    maximum_chromaticity_gain=1.35,
    maximum_white_mix=0.12,
    minimum_mean_channel_peak=0.48,
    minimum_p10_channel_peak=0.28,
)

_DARK_SCENE_CUES = (
    "暗调",
    "微光",
    "助眠",
    "夜晚",
    "夜间",
    "深夜",
    "隐约",
    "昏暗",
    "低照度",
    "神秘",
    "私密",
    "dim",
    "low light",
    "sleep",
    "night",
    "mysterious",
    "private",
)
_ENERGETIC_SCENE_CUES = (
    "霓虹",
    "能量",
    "活力",
    "健身",
    "运动",
    "涂鸦",
    "街头",
    "动态",
    "互动",
    "唤醒",
    "清醒",
    "快速",
    "即时吸引",
    "明亮",
    "neon",
    "energy",
    "energetic",
    "vibrant",
    "fitness",
    "gym",
    "sport",
    "graffiti",
    "street",
    "dynamic",
    "interactive",
    "wake",
    "bright",
)


def scene_brightness_policy(scene: str) -> BrightnessPolicy:
    """Choose a luminance floor while honoring explicit dark-scene intent."""

    normalized = " ".join(str(scene).casefold().split())
    if any(cue in normalized for cue in _DARK_SCENE_CUES):
        return DARK_BRIGHTNESS_POLICY
    if any(cue in normalized for cue in _ENERGETIC_SCENE_CUES):
        return ENERGETIC_BRIGHTNESS_POLICY
    return STANDARD_BRIGHTNESS_POLICY


def _luminance_summary(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = relative_luminance_rgb8(rgb)
    channel_peak = np.max(rgb / 255.0, axis=2)
    return {
        "mean_luminance": float(luminance.mean()),
        "p10_luminance": float(np.quantile(luminance, 0.10)),
        "below_0_20_fraction": float(np.mean(luminance < 0.20)),
        "mean_channel_peak": float(channel_peak.mean()),
        "p10_channel_peak": float(np.quantile(channel_peak, 0.10)),
    }


def apply_scene_brightness_floor(
    image: Image.Image,
    scene: str,
) -> tuple[Image.Image, dict[str, object]]:
    """Brighten a light field without rerunning diffusion or changing its layout."""

    source = image.convert("RGB")
    policy = scene_brightness_policy(scene)
    before = _luminance_summary(source)
    low_luminance = (
        before["mean_luminance"] < policy.target_mean_luminance
        or before["p10_luminance"] < policy.target_p10_luminance
    )
    low_channel_energy = (
        before["mean_channel_peak"] < policy.minimum_mean_channel_peak
        or before["p10_channel_peak"] < policy.minimum_p10_channel_peak
    )
    # Relative luminance assigns very different weights to red, green, and blue.
    # Requiring both low luminance and low channel energy prevents saturated red or
    # blue from being whitened merely because of its hue.
    needs_lift = low_luminance and low_channel_energy
    if not needs_lift:
        return source, {
            "applied": False,
            "policy": asdict(policy),
            "before": before,
            "after": before,
            "requested_chromaticity_gain": 1.0,
            "mean_effective_chromaticity_gain": 1.0,
            "white_mix": 0.0,
        }

    requested_gain = max(
        1.0,
        policy.target_mean_luminance / max(before["mean_luminance"], 1e-6),
        policy.target_p10_luminance / max(before["p10_luminance"], 1e-6),
    )
    requested_gain = min(requested_gain, policy.maximum_chromaticity_gain)
    source_rgb = np.asarray(source, dtype=np.float32)
    gained_rgb, effective_gain = apply_chromaticity_preserving_gain(
        source_rgb,
        np.full((*source_rgb.shape[:2], 1), requested_gain, dtype=np.float32),
    )
    gained = Image.fromarray(gained_rgb, mode="RGB")
    after_gain = _luminance_summary(gained)

    mean_mix = max(
        0.0,
        (policy.target_mean_luminance - after_gain["mean_luminance"])
        / max(1.0 - after_gain["mean_luminance"], 1e-6),
    )
    p10_mix = max(
        0.0,
        (policy.target_p10_luminance - after_gain["p10_luminance"])
        / max(1.0 - after_gain["p10_luminance"], 1e-6),
    )
    still_low_channel_energy = (
        after_gain["mean_channel_peak"] < policy.minimum_mean_channel_peak
        or after_gain["p10_channel_peak"] < policy.minimum_p10_channel_peak
    )
    white_mix = (
        min(max(mean_mix, p10_mix), policy.maximum_white_mix)
        if still_low_channel_energy
        else 0.0
    )
    if white_mix > 0.0:
        result_rgb = blend_with_white_linear(gained_rgb, white_mix)
        result = Image.fromarray(result_rgb, mode="RGB")
    else:
        result = gained
    after = _luminance_summary(result)
    return result, {
        "applied": True,
        "policy": asdict(policy),
        "before": before,
        "after_gain": after_gain,
        "after": after,
        "requested_chromaticity_gain": float(requested_gain),
        "mean_effective_chromaticity_gain": float(np.mean(effective_gain)),
        "white_mix": float(white_mix),
    }


@dataclass(frozen=True, slots=True)
class GradientStop:
    position: float
    rgb: tuple[int, int, int]
    color_name: str
    role: str


@dataclass(frozen=True, slots=True)
class StructuredGradientPlan:
    direction: str
    stops: tuple[GradientStop, ...]
    dominant_color: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "stops": [asdict(stop) for stop in self.stops],
            "dominant_color": self.dominant_color,
            "source": self.source,
        }


def _prompt_colors(prompt: str) -> list[tuple[int, str, tuple[int, int, int], bool]]:
    """Return non-overlapping named colors in textual order."""

    lowered = prompt.casefold()
    matches: list[tuple[int, str, tuple[int, int, int], bool]] = []
    for start, end, color_name, rgb in matched_color_spans(lowered):
        prefix = lowered[max(0, start - 28) : start]
        suffix = lowered[end : min(len(lowered), end + 32)]
        dominant = bool(
            re.search(
                r"\b(?:dominant|primary|main)(?:\s+color)?\s*$",
                prefix,
            )
            or re.match(
                r"\s+(?:as\s+)?(?:the\s+)?(?:dominant|primary|main)\b",
                suffix,
            )
        )
        matches.append((start, color_name, rgb, dominant))
    return matches


def _fallback_colors_from_image(image: Image.Image) -> list[tuple[str, tuple[int, int, int]]]:
    source = image.convert("RGB")
    profile = source.resize((source.width, 1), Image.Resampling.BILINEAR).filter(
        ImageFilter.GaussianBlur(radius=max(4.0, source.width * 0.08))
    )
    array = np.asarray(profile, dtype=np.uint8)[0]
    return [
        (f"source_{label}", tuple(int(value) for value in array[index]))
        for label, index in (
            ("left", 0),
            ("center", len(array) // 2),
            ("right", len(array) - 1),
        )
    ]


def build_structured_gradient_plan(
    prompt: str,
    source_image: Image.Image,
) -> StructuredGradientPlan:
    """Build a maximum-four-stop horizontal plan from explicit prompt colors."""

    unsupported = unsupported_color_terms(prompt)
    if unsupported:
        raise ValueError(
            "prompt contains colors that the structured renderer cannot parse: "
            + ", ".join(unsupported)
        )
    mentions = _prompt_colors(prompt)
    anchors = [
        anchor for anchor in extract_color_anchors(prompt) if anchor.axis == "horizontal"
    ]
    grouped: dict[float, list[object]] = {}
    for anchor in anchors:
        grouped.setdefault(anchor.position, []).append(anchor)

    if len(grouped) >= 2:
        stops: list[GradientStop] = []
        for position in sorted(grouped):
            values = grouped[position]
            weights = np.asarray([anchor.weight for anchor in values], dtype=np.float32)
            colors = np.asarray([anchor.rgb for anchor in values], dtype=np.float32)
            rgb = tuple(
                int(value)
                for value in np.rint(np.average(colors, axis=0, weights=weights))
            )
            name = "+".join(anchor.color_name for anchor in values)
            role = (
                "primary"
                if any(
                    mention_name in name and dominant
                    for _offset, mention_name, _rgb, dominant in mentions
                )
                else "secondary"
            )
            stops.append(GradientStop(float(position), rgb, name, role))
        if stops[0].position > 0.0:
            stops.insert(0, GradientStop(0.0, stops[0].rgb, stops[0].color_name, stops[0].role))
        if stops[-1].position < 1.0:
            stops.append(
                GradientStop(1.0, stops[-1].rgb, stops[-1].color_name, stops[-1].role)
            )
        stops = stops[:4] if len(stops) > 4 else stops
        dominant = next(
            (stop.color_name for stop in stops if stop.role == "primary"),
            stops[-1].color_name,
        )
        return StructuredGradientPlan(
            direction="horizontal",
            stops=tuple(stops),
            dominant_color=dominant,
            source="prompt_spatial_anchors",
        )

    unique_mentions: list[tuple[str, tuple[int, int, int], bool]] = []
    seen_names: set[str] = set()
    for _offset, name, rgb, dominant in mentions:
        if name not in seen_names:
            unique_mentions.append((name, rgb, dominant))
            seen_names.add(name)

    if not unique_mentions:
        fallback = _fallback_colors_from_image(source_image)
        return StructuredGradientPlan(
            direction="horizontal",
            stops=tuple(
                GradientStop(position, rgb, name, "secondary")
                for position, (name, rgb) in zip((0.0, 0.5, 1.0), fallback)
            ),
            dominant_color=fallback[1][0],
            source="diffusion_column_profile",
        )

    dominant_item = next((item for item in unique_mentions if item[2]), None)
    if len(unique_mentions) == 1:
        name, rgb, _dominant = unique_mentions[0]
        return StructuredGradientPlan(
            direction="horizontal",
            stops=(
                GradientStop(0.0, rgb, name, "primary"),
                GradientStop(1.0, rgb, name, "primary"),
            ),
            dominant_color=name,
            source="single_prompt_color",
        )

    if dominant_item is not None:
        secondary = next(item for item in unique_mentions if item is not dominant_item)
        name, rgb, _dominant = dominant_item
        stops = [
            GradientStop(0.0, secondary[1], secondary[0], "secondary"),
            GradientStop(0.42, rgb, name, "primary"),
            GradientStop(1.0, rgb, name, "primary"),
        ]
        return StructuredGradientPlan(
            direction="horizontal",
            stops=tuple(stops),
            dominant_color=name,
            source="prompt_dominant_color",
        )

    selected = unique_mentions[:4]
    positions = {
        2: (0.0, 1.0),
        3: (0.0, 0.5, 1.0),
        4: (0.0, 0.34, 0.68, 1.0),
    }[len(selected)]
    stops = tuple(
        GradientStop(position, rgb, name, "primary" if index == 1 else "secondary")
        for index, (position, (name, rgb, _dominant)) in enumerate(
            zip(positions, selected)
        )
    )
    return StructuredGradientPlan(
        direction="horizontal",
        stops=stops,
        dominant_color=stops[1].color_name,
        source="prompt_color_order",
    )


def render_base_gradient(
    plan: StructuredGradientPlan,
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Render a vertically uniform RGB gradient from the structured plan."""

    positions = np.asarray([stop.position for stop in plan.stops], dtype=np.float32)
    colors = np.asarray([stop.rgb for stop in plan.stops], dtype=np.float32)
    coordinate = np.linspace(0.0, 1.0, width, dtype=np.float32)
    line = np.stack(
        [
            np.interp(coordinate, positions, colors[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    pixels = np.repeat(line[np.newaxis, :, :], height, axis=0)
    return Image.fromarray(np.rint(np.clip(pixels, 0, 255)).astype(np.uint8), mode="RGB")


def structured_gradient_metrics(image: Image.Image) -> dict[str, float]:
    """Measure organizer-style horizontal structure and restrained saturation."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    column_profile = rgb.mean(axis=0, keepdims=True)
    residual = rgb - column_profile
    total_variance = float(np.var(rgb))
    residual_variance = float(np.var(residual))
    horizontal_explained = (
        1.0
        if total_variance <= 1e-12
        else float(np.clip(1.0 - residual_variance / total_variance, 0.0, 1.0))
    )
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 1e-8,
    )
    return {
        "vertical_color_variation": float(np.mean(np.abs(residual))),
        "horizontal_structure_explained": horizontal_explained,
        "mean_saturation": float(saturation.mean()),
    }


def _render_with_luminance_texture(
    base: Image.Image,
    source_image: Image.Image,
    *,
    strength: float,
) -> Image.Image:
    if strength == 0.0:
        return base.copy()
    source = source_image.convert("RGB").resize(base.size, Image.Resampling.BILINEAR)
    raw = np.asarray(source, dtype=np.float32) / 255.0
    luminance = np.sum(
        raw * np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
        axis=2,
    )
    low_pass = Image.fromarray(
        np.rint(luminance * 255.0).astype(np.uint8),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=max(6.0, base.height * 0.12)))
    field = np.asarray(low_pass, dtype=np.float32) / 255.0
    centered = field - float(field.mean())
    scale = max(float(np.quantile(np.abs(centered), 0.95)), 1e-6)
    normalized = np.clip(centered / scale, -1.0, 1.0)
    # At the recommended strength 0.10, LoRA contributes at most ±4.5% gain.
    gain = 1.0 + normalized[:, :, None] * (0.45 * strength)
    base_rgb = np.asarray(base, dtype=np.float32)
    result, _effective_gain = apply_chromaticity_preserving_gain(base_rgb, gain)
    return Image.fromarray(result, mode="RGB")


def render_structured_gradient(
    source_image: Image.Image,
    prompt: str,
    *,
    texture_strength: float = DEFAULT_TEXTURE_STRENGTH,
) -> tuple[Image.Image, dict[str, object]]:
    """Render a prompt-faithful gradient and safely retain weak LoRA luminance."""

    if not 0.0 <= texture_strength <= MAX_TEXTURE_STRENGTH:
        raise ValueError(
            f"texture_strength must be from 0 to {MAX_TEXTURE_STRENGTH:g}"
        )
    source = source_image.convert("RGB")
    plan = build_structured_gradient_plan(prompt, source)
    base = render_base_gradient(plan, width=source.width, height=source.height)
    requested = float(texture_strength)
    candidates = []
    for strength in (requested, min(requested, 0.08), min(requested, 0.04), 0.0):
        if strength not in candidates:
            candidates.append(strength)

    attempts: list[dict[str, object]] = []
    rendered = base
    effective = 0.0
    for strength in candidates:
        candidate = _render_with_luminance_texture(
            base,
            source,
            strength=strength,
        )
        metrics = structured_gradient_metrics(candidate)
        accepted = metrics["vertical_color_variation"] <= MAX_VERTICAL_VARIATION
        attempts.append(
            {
                "texture_strength": float(strength),
                "accepted": bool(accepted),
                **metrics,
            }
        )
        if accepted:
            rendered = candidate
            effective = float(strength)
            break

    base_metrics = structured_gradient_metrics(base)
    final_metrics = structured_gradient_metrics(rendered)
    return rendered, {
        "applied": True,
        "render_mode": "structured_horizontal_gradient_with_lora_luminance",
        "plan": plan.to_dict(),
        "requested_texture_strength": requested,
        "effective_texture_strength": effective,
        "maximum_luminance_gain_deviation": 0.45 * effective,
        "fallback_attempts": attempts,
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "quality_status": "accepted",
        # Compatibility fields consumed by the existing report and UI.
        "strength": effective,
        "requested_strength": requested,
        "effective_strength": effective,
        "pre_guidance_layout_error": None,
        "post_guidance_layout_error": None,
        "pre_guidance_anchor_color_error": None,
        "post_guidance_anchor_color_error": None,
        "anchors": [
            {
                "position": stop.position,
                "rgb": list(stop.rgb),
                "color_name": stop.color_name,
                "role": stop.role,
                "axis": "horizontal",
            }
            for stop in plan.stops
        ],
    }
