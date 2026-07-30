"""Vectorized D65 sRGB, CIE XYZ, xyY and CIE Lab conversions."""

from __future__ import annotations

import numpy as np


D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
XYZ_TO_RGB = np.linalg.inv(RGB_TO_XYZ)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert unit-range nonlinear sRGB values to linear RGB."""

    values = np.asarray(srgb, dtype=np.float64)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB to clipped unit-range nonlinear sRGB."""

    values = np.asarray(linear_rgb, dtype=np.float64)
    srgb = np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(np.maximum(values, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def srgb_to_xyz(srgb: np.ndarray) -> np.ndarray:
    """Convert unit-range nonlinear sRGB values to D65 CIE XYZ."""

    linear = srgb_to_linear(srgb)
    return np.matmul(linear, RGB_TO_XYZ.T)


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """Convert D65 CIE XYZ values to clipped unit-range nonlinear sRGB."""

    linear = np.matmul(np.asarray(xyz, dtype=np.float64), XYZ_TO_RGB.T)
    return linear_to_srgb(linear)


def xyz_to_xyy(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to xyY; zero-energy colors receive x=y=0."""

    values = np.asarray(xyz, dtype=np.float64)
    total = values[..., 0] + values[..., 1] + values[..., 2]
    x = np.divide(
        values[..., 0],
        total,
        out=np.zeros_like(total),
        where=total > 1e-12,
    )
    y = np.divide(
        values[..., 1],
        total,
        out=np.zeros_like(total),
        where=total > 1e-12,
    )
    return np.stack((x, y, values[..., 1]), axis=-1)


def xyy_to_xyz(xyy: np.ndarray) -> np.ndarray:
    """Convert xyY to XYZ; entries with y=0 become zero."""

    values = np.asarray(xyy, dtype=np.float64)
    x, y, luminance = values[..., 0], values[..., 1], values[..., 2]
    scale = np.divide(
        luminance,
        y,
        out=np.zeros_like(luminance),
        where=np.abs(y) > 1e-12,
    )
    xyz_x = x * scale
    xyz_z = (1.0 - x - y) * scale
    return np.stack((xyz_x, luminance, xyz_z), axis=-1)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Convert D65 XYZ to CIE Lab."""

    scaled = np.asarray(xyz, dtype=np.float64) / D65_WHITE
    delta = 6.0 / 29.0
    transformed = np.where(
        scaled > delta**3,
        np.cbrt(scaled),
        scaled / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lightness = 116.0 * transformed[..., 1] - 16.0
    a = 500.0 * (transformed[..., 0] - transformed[..., 1])
    b = 200.0 * (transformed[..., 1] - transformed[..., 2])
    return np.stack((lightness, a, b), axis=-1)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """Convert CIE Lab to D65 XYZ."""

    values = np.asarray(lab, dtype=np.float64)
    fy = (values[..., 0] + 16.0) / 116.0
    fx = fy + values[..., 1] / 500.0
    fz = fy - values[..., 2] / 200.0
    transformed = np.stack((fx, fy, fz), axis=-1)
    delta = 6.0 / 29.0
    scaled = np.where(
        transformed > delta,
        transformed**3,
        3.0 * delta**2 * (transformed - 4.0 / 29.0),
    )
    return scaled * D65_WHITE


def rgb8_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8-like RGB values to D65 XYZ."""

    return srgb_to_xyz(np.asarray(rgb, dtype=np.float64) / 255.0)


def rgb8_to_xyy(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8-like RGB values to CIE xyY."""

    return xyz_to_xyy(rgb8_to_xyz(rgb))


def rgb8_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8-like RGB values to CIE Lab."""

    return xyz_to_lab(rgb8_to_xyz(rgb))


def lab_to_rgb8(lab: np.ndarray) -> np.ndarray:
    """Convert CIE Lab values to clipped uint8 sRGB."""

    srgb = xyz_to_srgb(lab_to_xyz(lab))
    return np.rint(srgb * 255.0).astype(np.uint8)


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """Return D65 relative luminance Y for uint8-like RGB values."""

    return rgb8_to_xyz(rgb)[..., 1]
