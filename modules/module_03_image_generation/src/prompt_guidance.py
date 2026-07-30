"""Low-frequency color anchoring for prompt-faithful panoramic light textures."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from PIL import Image

from modules.color_vocabulary import COLOR_RGB

POSITION_TERMS: tuple[tuple[str, str, float], ...] = (
    ("upper", "vertical", 0.0),
    ("top", "vertical", 0.0),
    ("horizon", "vertical", 0.5),
    ("middle", "vertical", 0.5),
    ("lower", "vertical", 1.0),
    ("bottom", "vertical", 1.0),
    ("left", "horizontal", 0.0),
    ("center", "horizontal", 0.5),
    ("central", "horizontal", 0.5),
    ("right", "horizontal", 1.0),
)


@dataclass(frozen=True, slots=True)
class ColorAnchor:
    axis: str
    position: float
    rgb: tuple[int, int, int]
    color_name: str
    position_name: str
    weight: float
    scope: str


def _anchor_weight_and_scope(text: str, start: int, end: int) -> tuple[float, str]:
    clause_start = max(text.rfind(",", 0, start), text.rfind(".", 0, start)) + 1
    comma_end = text.find(",", end)
    period_end = text.find(".", end)
    candidates = [value for value in (comma_end, period_end) if value >= 0]
    clause_end = min(candidates) if candidates else len(text)
    clause = text[clause_start:clause_end]
    if re.search(r"\b(?:center|central|localized|focal)\b", clause) or re.search(
        r"\b(?:cloud-like|cloudlike)\s+diffusion\b",
        clause,
    ):
        return 0.28, "local"
    if re.search(
        r"\b(?:across|throughout|area|broad|both sides|edges)\b",
        clause,
    ):
        return 1.55, "broad"
    return 1.0, "standard"


def extract_color_anchors(prompt: str, *, max_distance: int = 36) -> list[ColorAnchor]:
    """Associate named colors with the nearest explicit spatial term."""

    lowered = prompt.casefold()
    positions: list[tuple[int, str, str, float]] = []
    for term, axis, value in POSITION_TERMS:
        positions.extend(
            (match.start(), term, axis, value)
            for match in re.finditer(rf"\b{re.escape(term)}\b", lowered)
        )

    anchors: list[ColorAnchor] = []
    occupied_spans: list[tuple[int, int]] = []
    for color_name, rgb in COLOR_RGB:
        for match in re.finditer(rf"\b{re.escape(color_name)}\b", lowered):
            if any(start <= match.start() < end for start, end in occupied_spans):
                continue
            following = [
                item
                for item in positions
                if 0 <= item[0] - match.end() <= max_distance
            ]
            nearest = (
                min(following, key=lambda item: item[0] - match.end())
                if following
                else min(
                    positions,
                    key=lambda item: abs(item[0] - match.start()),
                    default=None,
                )
            )
            if nearest is None or abs(nearest[0] - match.start()) > max_distance:
                continue
            _offset, position_name, axis, value = nearest
            weight, scope = _anchor_weight_and_scope(
                lowered,
                match.start(),
                match.end(),
            )
            anchors.append(
                ColorAnchor(
                    axis=axis,
                    position=value,
                    rgb=rgb,
                    color_name=color_name,
                    position_name=position_name,
                    weight=weight,
                    scope=scope,
                )
            )
            occupied_spans.append(match.span())
    return anchors


def _axis_guide(
    anchors: list[ColorAnchor],
    *,
    width: int,
    height: int,
) -> np.ndarray | None:
    by_axis = {
        axis: [anchor for anchor in anchors if anchor.axis == axis]
        for axis in ("vertical", "horizontal")
    }
    usable = [
        (axis, values)
        for axis, values in by_axis.items()
        if len({anchor.position for anchor in values}) >= 2
    ]
    if not usable:
        positioned = [anchor for anchor in anchors if anchor.scope != "local"]
        if not positioned:
            return None
        dominant = max(positioned, key=lambda anchor: anchor.weight)
        color = np.asarray(dominant.rgb, dtype=np.float32)
        return np.broadcast_to(color, (height, width, 3)).copy()
    axis, selected = max(usable, key=lambda item: len(item[1]))

    grouped: dict[float, list[tuple[np.ndarray, float]]] = {}
    for anchor in selected:
        grouped.setdefault(anchor.position, []).append(
            (np.asarray(anchor.rgb, dtype=np.float32), anchor.weight)
        )
    positions = np.asarray(sorted(grouped), dtype=np.float32)
    colors = np.asarray(
        [
            np.average(
                np.asarray([item[0] for item in grouped[position]]),
                axis=0,
                weights=np.asarray([item[1] for item in grouped[position]]),
            )
            for position in positions
        ],
        dtype=np.float32,
    )
    coordinate = np.linspace(0.0, 1.0, height if axis == "vertical" else width)
    line = np.stack(
        [np.interp(coordinate, positions, colors[:, channel]) for channel in range(3)],
        axis=-1,
    )
    if axis == "vertical":
        return np.repeat(line[:, np.newaxis, :], width, axis=1)
    return np.repeat(line[np.newaxis, :, :], height, axis=0)


def _broad_anchor_color_error(
    rgb: np.ndarray,
    anchors: list[ColorAnchor],
) -> float | None:
    errors: list[float] = []
    height, width = rgb.shape[:2]
    for anchor in anchors:
        if anchor.scope == "local":
            continue
        if anchor.axis == "vertical":
            center = int(round(anchor.position * (height - 1)))
            radius = max(1, height // 6)
            region = rgb[max(0, center - radius) : min(height, center + radius + 1)]
        else:
            center = int(round(anchor.position * (width - 1)))
            radius = max(1, width // 6)
            region = rgb[:, max(0, center - radius) : min(width, center + radius + 1)]
        mean = region.mean(axis=(0, 1))
        target = np.asarray(anchor.rgb, dtype=np.float32)
        errors.append(float(np.mean(np.abs(mean - target)) / 255.0))
    return float(np.mean(errors)) if errors else None


def apply_prompt_color_guidance(
    image: Image.Image,
    prompt: str,
    *,
    strength: float = 0.64,
) -> tuple[Image.Image, dict[str, object]]:
    """Adaptively blend a spatial guide only as strongly as the Raw image needs."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be from 0 to 1")
    source = image.convert("RGB")
    anchors = extract_color_anchors(prompt)
    anchor_report = [
        {
            "axis": anchor.axis,
            "position": anchor.position,
            "rgb": list(anchor.rgb),
            "color_name": anchor.color_name,
            "position_name": anchor.position_name,
            "weight": anchor.weight,
            "scope": anchor.scope,
        }
        for anchor in anchors
    ]
    guide = _axis_guide(anchors, width=source.width, height=source.height)
    if guide is None or strength == 0:
        return source, {
            "applied": False,
            "strength": strength,
            "requested_strength": strength,
            "effective_strength": 0.0,
            "pre_guidance_layout_error": None,
            "post_guidance_layout_error": None,
            "pre_guidance_anchor_color_error": None,
            "post_guidance_anchor_color_error": None,
            "anchors": anchor_report,
        }

    source_array = np.asarray(source, dtype=np.float32)
    pre_anchor_error = _broad_anchor_color_error(source_array, anchors)
    pre_error = float(np.mean(np.abs(source_array - guide)) / 255.0)
    # Ignore tiny palette differences and reach the requested maximum only
    # when the low-frequency layout is substantially wrong.
    mismatch = float(np.clip((pre_error - 0.03) / 0.20, 0.0, 1.0))
    effective_strength = strength * mismatch
    combined = np.clip(
        source_array * (1.0 - effective_strength) + guide * effective_strength,
        0,
        255,
    ).astype(np.uint8)
    post_error = float(np.mean(np.abs(combined - guide)) / 255.0)
    post_anchor_error = _broad_anchor_color_error(combined, anchors)
    return Image.fromarray(combined, mode="RGB"), {
        "applied": bool(effective_strength > 0.0),
        # Keep this compatibility field as the strength actually applied.
        "strength": effective_strength,
        "requested_strength": strength,
        "effective_strength": effective_strength,
        "pre_guidance_layout_error": pre_error,
        "post_guidance_layout_error": post_error,
        "pre_guidance_anchor_color_error": pre_anchor_error,
        "post_guidance_anchor_color_error": post_anchor_error,
        "anchors": anchor_report,
    }
