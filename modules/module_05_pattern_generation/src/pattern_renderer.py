"""Deterministic low-frequency thematic light-field enhancement."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image, ImageFilter

from modules.linear_luminance import (
    apply_chromaticity_preserving_gain,
    relative_luminance_rgb8,
)

from .theme_extractor import PatternAttributes


def _stable_phase(theme: str, seed: int) -> float:
    digest = hashlib.sha256(f"{theme}:{int(seed)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 * math.tau


def _coordinate_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    return np.meshgrid(x, y)


def _field(
    width: int,
    height: int,
    attributes: PatternAttributes,
    *,
    seed: int,
) -> np.ndarray:
    x, y = _coordinate_grid(width, height)
    phase = _stable_phase(attributes.theme, seed)
    if attributes.motif == "flowing":
        diagonal = x * 0.82 + y * 0.38
        values = 0.5 + 0.5 * np.sin(diagonal * math.pi * 1.15 + phase)
    elif attributes.motif == "radiant":
        center_x = 0.16 * math.sin(phase)
        center_y = 0.12 * math.cos(phase)
        radius = ((x - center_x) / 1.15) ** 2 + ((y - center_y) / 0.82) ** 2
        values = np.exp(-radius * 1.45)
    elif attributes.motif == "breathing":
        horizontal = np.exp(-((x - 0.12 * math.sin(phase)) / 0.92) ** 2)
        vertical = 0.86 + 0.14 * np.cos(y * math.pi)
        values = horizontal * vertical
    else:
        raise ValueError(f"unsupported motif: {attributes.motif}")
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros((height, width), dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def render_pattern(
    base_image: Image.Image,
    attributes: PatternAttributes,
    *,
    seed: int,
) -> tuple[Image.Image, dict[str, float | int | str | bool]]:
    """Apply a smooth emitted-light modulation without drawing discrete objects."""

    source = base_image.convert("RGB")
    if attributes.pattern_strength == 0:
        return source.copy(), {
            "applied": False,
            "motif": attributes.motif,
            "seed": int(seed),
            "effective_strength": 0.0,
            "mean_absolute_change": 0.0,
            "mean_luminance_change": 0.0,
        }

    width, height = source.size
    field = _field(width, height, attributes, seed=seed)
    field_image = Image.fromarray(
        np.rint(field * 255.0).astype(np.uint8),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=max(4.0, height * 0.08)))
    modulation = np.asarray(field_image, dtype=np.float32)[:, :, None] / 255.0

    before = np.asarray(source, dtype=np.float32)
    centered = modulation - float(modulation.mean())
    maximum_deviation = max(float(np.max(np.abs(centered))), 1e-6)
    normalized = centered / maximum_deviation
    # Multiplying all RGB channels by the same smooth gain preserves local
    # chromaticity. Module five changes rhythm and luminance, never palette.
    # Keep module-five motion visibly subordinate to the organizer-faithful
    # gradient. Even at the UI maximum (0.18), luminance moves by at most 5%.
    gain = 1.0 + attributes.pattern_strength * 0.28 * normalized
    after_rgb, effective_gain = apply_chromaticity_preserving_gain(before, gain)
    after = after_rgb.astype(np.float32)
    rendered = Image.fromarray(after_rgb, mode="RGB")

    before_luminance = relative_luminance_rgb8(before)
    after_luminance = relative_luminance_rgb8(after)
    return rendered, {
        "applied": True,
        "motif": attributes.motif,
        "seed": int(seed),
        "effective_strength": float(attributes.pattern_strength),
        "field_blur_radius": round(max(4.0, height * 0.08), 3),
        "color_mode": "chromaticity_preserving_linear_rgb_gain",
        "maximum_gain_deviation": float(attributes.pattern_strength * 0.28),
        "maximum_effective_gain_deviation": float(
            np.max(np.abs(effective_gain - 1.0))
        ),
        "mean_absolute_change": float(np.mean(np.abs(before - after)) / 255.0),
        "mean_luminance_change": float(
            np.mean(np.abs(before_luminance - after_luminance))
        ),
    }
