"""Convert physical fixture proportions into diffusion image dimensions."""

from __future__ import annotations


def round_to_multiple(value: float, multiple: int = 8) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return max(multiple, int(round(value / multiple)) * multiple)


def dimensions_from_fixture(
    width_mm: float,
    height_mm: float,
    *,
    long_edge: int = 1024,
    multiple: int = 8,
) -> tuple[int, int]:
    """Fit a physical aspect ratio inside a diffusion-friendly long edge."""

    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("fixture dimensions must be positive")
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if long_edge < 64 or long_edge % multiple:
        raise ValueError("long_edge must be at least 64 and divisible by multiple")

    ratio = width_mm / height_mm
    if ratio >= 1:
        width = long_edge
        height = round_to_multiple(long_edge / ratio, multiple)
    else:
        height = long_edge
        width = round_to_multiple(long_edge * ratio, multiple)
    if min(width, height) < 64:
        raise ValueError(
            "fixture aspect ratio is too extreme to produce both image dimensions "
            "at least 64 pixels"
        )
    return width, height
