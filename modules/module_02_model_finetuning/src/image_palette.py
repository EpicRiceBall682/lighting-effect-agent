"""Pixel-level palette checks for generated light-effect training images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True, slots=True)
class PaletteAudit:
    """Summarize pixels outside the organizer's bright lighting palette."""

    forbidden_hue_fraction: float
    dark_pixel_fraction: float
    isolated_chroma_fraction: float
    allowed: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def audit_rgb_pixels(
    image: np.ndarray,
    *,
    max_forbidden_hue_fraction: float = 0.005,
    max_dark_pixel_fraction: float = 0.01,
    max_isolated_chroma_fraction: float = 0.005,
) -> PaletteAudit:
    """Check palette compliance and reject localized chroma artifacts."""

    rgb8 = np.asarray(image)
    if rgb8.ndim != 3 or rgb8.shape[2] < 3:
        raise ValueError("image must be an RGB-like array")
    rgb8 = np.clip(rgb8[:, :, :3], 0, 255).astype(np.uint8)
    rgb = rgb8.astype(np.float32) / 255.0

    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)

    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red_max = (maximum == red) & nonzero
    green_max = (maximum == green) & nonzero
    blue_max = (maximum == blue) & nonzero
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6)
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4
    hue /= 6.0

    # 0.19–0.52 spans yellow-green, green and cyan. Low-saturation ivory/white
    # pixels are ignored because their hue is visually insignificant.
    # A 0.12 saturation floor still ignores nearly neutral ivory/white, while
    # catching the visibly pale green band produced by blue-to-yellow blends.
    forbidden_hue = (saturation >= 0.12) & (hue >= 0.19) & (hue <= 0.52)

    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = 0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2]
    dark_pixels = luminance < 0.20

    if min(rgb8.shape[:2]) < 8:
        isolated_fraction = 0.0
    else:
        radius = max(2.0, min(rgb8.shape[:2]) * 0.035)
        low_pass = np.asarray(
            Image.fromarray(rgb8, mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=radius)
            ),
            dtype=np.float32,
        ) / 255.0
        source_luma = (
            0.2126 * rgb[:, :, 0]
            + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]
        )
        low_luma = (
            0.2126 * low_pass[:, :, 0]
            + 0.7152 * low_pass[:, :, 1]
            + 0.0722 * low_pass[:, :, 2]
        )
        chroma = rgb - source_luma[:, :, None]
        low_chroma = low_pass - low_luma[:, :, None]
        chroma_residual = np.sqrt(
            np.mean((chroma - low_chroma) ** 2, axis=2)
        )
        isolated_fraction = float((chroma_residual > 0.03).mean())

    forbidden_fraction = float(forbidden_hue.mean())
    dark_fraction = float(dark_pixels.mean())
    return PaletteAudit(
        forbidden_hue_fraction=forbidden_fraction,
        dark_pixel_fraction=dark_fraction,
        isolated_chroma_fraction=isolated_fraction,
        allowed=(
            forbidden_fraction <= max_forbidden_hue_fraction
            and dark_fraction <= max_dark_pixel_fraction
            and isolated_fraction <= max_isolated_chroma_fraction
        ),
    )


def audit_image_file(path: Path, **kwargs: float) -> PaletteAudit:
    with Image.open(path) as image:
        return audit_rgb_pixels(np.asarray(image.convert("RGB")), **kwargs)
