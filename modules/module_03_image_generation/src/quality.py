"""Lightweight quality diagnostics for generated raw light-effect images."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True, slots=True)
class ImageQualityReport:
    mean_luminance: float
    near_black_fraction: float
    forbidden_hue_fraction: float
    abrupt_transition_fraction: float
    isolated_chroma_fraction: float
    broad_chroma_fraction: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ImageQualityPolicy:
    """Acceptance thresholds for a deliverable luminaire texture."""

    minimum_mean_luminance: float = 0.25
    maximum_near_black_fraction: float = 0.10
    # Retained for report compatibility. Full-hue generation is now allowed.
    maximum_forbidden_hue_fraction: float = 1.0
    maximum_abrupt_transition_fraction: float = 0.01
    maximum_isolated_chroma_fraction: float = 0.001
    maximum_broad_chroma_fraction: float = 0.003
    maximum_prompt_layout_error: float = 0.22
    maximum_prompt_anchor_color_error: float = 0.22

    def failures(
        self,
        report: ImageQualityReport,
        *,
        prompt_layout_error: float | None = None,
        prompt_anchor_color_error: float | None = None,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        checks = (
            (
                report.mean_luminance < self.minimum_mean_luminance,
                "mean_luminance",
                report.mean_luminance,
                self.minimum_mean_luminance,
                "minimum",
            ),
            (
                report.near_black_fraction > self.maximum_near_black_fraction,
                "near_black_fraction",
                report.near_black_fraction,
                self.maximum_near_black_fraction,
                "maximum",
            ),
            (
                report.forbidden_hue_fraction
                > self.maximum_forbidden_hue_fraction,
                "forbidden_hue_fraction",
                report.forbidden_hue_fraction,
                self.maximum_forbidden_hue_fraction,
                "maximum",
            ),
            (
                report.abrupt_transition_fraction
                > self.maximum_abrupt_transition_fraction,
                "abrupt_transition_fraction",
                report.abrupt_transition_fraction,
                self.maximum_abrupt_transition_fraction,
                "maximum",
            ),
            (
                report.isolated_chroma_fraction
                > self.maximum_isolated_chroma_fraction,
                "isolated_chroma_fraction",
                report.isolated_chroma_fraction,
                self.maximum_isolated_chroma_fraction,
                "maximum",
            ),
            (
                report.broad_chroma_fraction
                > self.maximum_broad_chroma_fraction,
                "broad_chroma_fraction",
                report.broad_chroma_fraction,
                self.maximum_broad_chroma_fraction,
                "maximum",
            ),
        )
        for failed, name, value, limit, direction in checks:
            if failed:
                failures.append(
                    f"{name}={value:.6f} violates its {direction} {limit:.6f}"
                )
        if (
            prompt_layout_error is not None
            and prompt_layout_error > self.maximum_prompt_layout_error
        ):
            failures.append(
                "prompt_layout_error="
                f"{prompt_layout_error:.6f} violates its maximum "
                f"{self.maximum_prompt_layout_error:.6f}"
            )
        if (
            prompt_anchor_color_error is not None
            and prompt_anchor_color_error
            > self.maximum_prompt_anchor_color_error
        ):
            failures.append(
                "prompt_anchor_color_error="
                f"{prompt_anchor_color_error:.6f} violates its maximum "
                f"{self.maximum_prompt_anchor_color_error:.6f}"
            )
        return tuple(failures)


def analyze_image_quality(image: Image.Image) -> ImageQualityReport:
    """Measure obvious failures while leaving final gamut mapping to module 4."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)

    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red_max = (maximum == red) & nonzero
    green_max = (maximum == green) & nonzero
    blue_max = (maximum == blue) & nonzero
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6)
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4
    hue /= 6.0

    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    forbidden_hue = (saturation >= 0.18) & (hue >= 0.19) & (hue <= 0.52)
    horizontal_difference = np.linalg.norm(np.diff(rgb, axis=1), axis=2)
    vertical_difference = np.linalg.norm(np.diff(rgb, axis=0), axis=2)
    abrupt_fraction = float(
        (
            np.count_nonzero(horizontal_difference > 0.18)
            + np.count_nonzero(vertical_difference > 0.18)
        )
        / (horizontal_difference.size + vertical_difference.size)
    )
    isolated_fraction = isolated_chroma_fraction(image)
    broad_fraction = broad_chroma_fraction(image)

    mean_luminance = float(luminance.mean())
    near_black_fraction = float((luminance < 0.02).mean())
    forbidden_fraction = float(forbidden_hue.mean())
    warnings: list[str] = []
    if mean_luminance < 0.25:
        warnings.append("image is much darker than the requested bright light-effect palette")
    if abrupt_fraction > 0.01:
        warnings.append("image contains many abrupt pixel transitions or edge artifacts")
    if isolated_fraction > 0.001:
        warnings.append("image contains localized chroma spots or speckle artifacts")
    if broad_fraction > 0.003:
        warnings.append("image contains broad local chroma blotches")
    return ImageQualityReport(
        mean_luminance=mean_luminance,
        near_black_fraction=near_black_fraction,
        forbidden_hue_fraction=forbidden_fraction,
        abrupt_transition_fraction=abrupt_fraction,
        isolated_chroma_fraction=isolated_fraction,
        broad_chroma_fraction=broad_fraction,
        warnings=tuple(warnings),
    )


def _chroma_residual(
    image: Image.Image,
    *,
    radius: float,
) -> np.ndarray:
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    low_pass = np.asarray(
        image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.float32,
    ) / 255.0
    weights = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    source_chroma = source - np.sum(source * weights, axis=2, keepdims=True)
    low_chroma = low_pass - np.sum(low_pass * weights, axis=2, keepdims=True)
    return np.sqrt(np.mean((source_chroma - low_chroma) ** 2, axis=2))


def isolated_chroma_fraction(
    image: Image.Image,
    *,
    threshold: float = 0.03,
) -> float:
    """Measure small chroma deviations against a broad local color field."""

    source = image.convert("RGB")
    if min(source.size) < 8:
        return 0.0
    radius = max(2.0, source.height * 0.035)
    return float((_chroma_residual(source, radius=radius) > threshold).mean())


def broad_chroma_fraction(
    image: Image.Image,
    *,
    threshold: float = 0.025,
) -> float:
    """Measure medium-to-large chroma blotches against a broad color field.

    The isolated detector intentionally ignores wide regions. A second,
    medium-scale low-pass radius catches irregular cloud-like patches while
    leaving broad, intentional panel gradients below the acceptance limit.
    """

    source = image.convert("RGB")
    if min(source.size) < 16:
        return 0.0
    radius = max(6.0, source.height * 0.075)
    return float((_chroma_residual(source, radius=radius) > threshold).mean())


def suppress_broad_chroma_artifacts(
    image: Image.Image,
    *,
    threshold: float = 0.025,
) -> tuple[Image.Image, dict[str, float | bool]]:
    """Locally smooth medium-scale chroma blotches and preserve the full gradient."""

    source = image.convert("RGB")
    if min(source.size) < 16:
        return source.copy(), {
            "applied": False,
            "detected_fraction": 0.0,
            "changed_pixel_fraction": 0.0,
            "remaining_broad_chroma_fraction": 0.0,
        }
    radius = max(6.0, source.height * 0.075)
    detected = _chroma_residual(source, radius=radius) > threshold
    detected_fraction = float(detected.mean())
    if detected_fraction == 0.0:
        return source.copy(), {
            "applied": False,
            "detected_fraction": 0.0,
            "changed_pixel_fraction": 0.0,
            "remaining_broad_chroma_fraction": broad_chroma_fraction(source),
        }

    expansion = max(9, int(round(source.height * 0.065)) | 1)
    cleaned = source.copy()
    for _pass in range(2):
        detected = _chroma_residual(cleaned, radius=radius) > threshold
        if not detected.any():
            break
        mask = Image.fromarray(
            np.where(detected, 255, 0).astype(np.uint8),
            mode="L",
        )
        feather = mask.filter(ImageFilter.MaxFilter(expansion)).filter(
            ImageFilter.GaussianBlur(radius=max(3.0, source.height * 0.025))
        )
        local_field = cleaned.filter(ImageFilter.GaussianBlur(radius=radius))
        cleaned = Image.composite(local_field, cleaned, feather)
    before = np.asarray(source, dtype=np.int16)
    after = np.asarray(cleaned, dtype=np.int16)
    changed = np.any(before != after, axis=2)
    return cleaned, {
        "applied": True,
        "detected_fraction": detected_fraction,
        "changed_pixel_fraction": float(changed.mean()),
        "remaining_broad_chroma_fraction": broad_chroma_fraction(cleaned),
    }


def suppress_isolated_chroma_artifacts(
    image: Image.Image,
    *,
    threshold: float = 0.03,
) -> tuple[Image.Image, dict[str, float | bool]]:
    """Feather only localized color residuals while preserving broad gradients."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    source = image.convert("RGB")
    before = np.asarray(source, dtype=np.float32)
    detection_radius = max(2.0, source.height * 0.035)
    residual = _chroma_residual(source, radius=detection_radius)
    hard_mask = residual > threshold

    if bool(np.any(hard_mask)):
        expansion = max(3, int(round(source.height * 0.05)))
        if expansion % 2 == 0:
            expansion += 1
        mask = Image.fromarray((hard_mask * 255).astype(np.uint8), mode="L")
        mask = mask.filter(ImageFilter.MaxFilter(expansion))
        mask = mask.filter(
            ImageFilter.GaussianBlur(radius=max(1.5, source.height * 0.02))
        )
        alpha = np.asarray(mask, dtype=np.float32)[:, :, None] / 255.0
        low_pass = np.asarray(
            source.filter(ImageFilter.GaussianBlur(radius=detection_radius)),
            dtype=np.float32,
        )
        cleaned_array = before * (1.0 - alpha) + low_pass * alpha
    else:
        cleaned_array = before

    cleaned = Image.fromarray(
        np.clip(np.rint(cleaned_array), 0, 255).astype(np.uint8),
        mode="RGB",
    )
    after = np.asarray(cleaned, dtype=np.float32)
    changed_fraction = float(
        (np.max(np.abs(after - before), axis=2) >= 2.0).mean()
    )
    return cleaned, {
        "applied": bool(changed_fraction > 0.0),
        "threshold": float(threshold),
        "base_softening_radius": 0.0,
        "detection_radius": float(detection_radius),
        "detected_fraction": float(hard_mask.mean()),
        "changed_pixel_fraction": changed_fraction,
        "mean_absolute_change": float(np.mean(np.abs(after - before)) / 255.0),
        "remaining_isolated_chroma_fraction": isolated_chroma_fraction(
            cleaned,
            threshold=threshold,
        ),
    }
