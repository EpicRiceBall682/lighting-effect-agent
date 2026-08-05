"""Linear-light helpers for changing brightness without shifting chromaticity."""

from __future__ import annotations

import numpy as np


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(np.maximum(values, 0.0), 1.0 / 2.4) - 0.055,
    )


def relative_luminance_rgb8(rgb: np.ndarray) -> np.ndarray:
    """Return D65 relative luminance for uint8-like sRGB pixels."""

    values = np.asarray(rgb, dtype=np.float32)
    if values.shape[-1] != 3:
        raise ValueError("rgb must end with three color channels")
    linear = _srgb_to_linear(np.clip(values / 255.0, 0.0, 1.0))
    return np.sum(
        linear * np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
        axis=-1,
    )


def blend_with_white_linear(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Lift sRGB pixels toward white in linear light with one uniform strength."""

    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    amount = float(strength)
    if not np.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ValueError("strength must be a finite value between zero and one")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("rgb must contain only finite values")

    linear = _srgb_to_linear(np.clip(values / 255.0, 0.0, 1.0))
    lifted = linear + (1.0 - linear) * amount
    encoded = np.clip(_linear_to_srgb(lifted), 0.0, 1.0)
    return np.rint(encoded * 255.0).astype(np.uint8)


def apply_chromaticity_preserving_gain(
    rgb: np.ndarray,
    gain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-pixel linear-light gain without clipping individual channels.

    Brightening is capped at the largest gain that keeps all three linear RGB
    channels inside the display gamut. This preserves each pixel's RGB direction
    instead of clipping one channel and shifting its hue.
    """

    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    requested = np.asarray(gain, dtype=np.float32)
    if requested.shape == values.shape[:2]:
        requested = requested[:, :, None]
    if requested.shape != (*values.shape[:2], 1):
        raise ValueError("gain must have shape (height, width) or (height, width, 1)")
    if not bool(np.all(np.isfinite(values))) or not bool(np.all(np.isfinite(requested))):
        raise ValueError("rgb and gain must contain only finite values")
    if bool(np.any(requested < 0.0)):
        raise ValueError("gain cannot be negative")

    srgb = np.clip(values / 255.0, 0.0, 1.0)
    linear = _srgb_to_linear(srgb)
    maximum = linear.max(axis=2, keepdims=True)
    maximum_gain = np.divide(
        1.0,
        maximum,
        out=np.full_like(maximum, np.inf),
        where=maximum > 1e-12,
    )
    effective_gain = np.minimum(requested, maximum_gain)
    adjusted = np.clip(linear * effective_gain, 0.0, 1.0)
    encoded = np.clip(_linear_to_srgb(adjusted), 0.0, 1.0)
    output = np.rint(encoded * 255.0).astype(np.uint8)
    return output, effective_gain[:, :, 0]
