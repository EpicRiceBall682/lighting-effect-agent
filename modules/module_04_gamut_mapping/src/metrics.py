"""Objective quality and compliance metrics for SDL gamut mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .color_spaces import relative_luminance, rgb8_to_lab
from .sdl_palette import SDLPalette


def _abrupt_transition_fraction(rgb: np.ndarray, threshold: float = 0.18) -> float:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    horizontal = np.linalg.norm(np.diff(values, axis=1), axis=2)
    vertical = np.linalg.norm(np.diff(values, axis=0), axis=2)
    count = np.count_nonzero(horizontal > threshold) + np.count_nonzero(vertical > threshold)
    return float(count / (horizontal.size + vertical.size))


def _box_blur(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
    total = np.zeros_like(values, dtype=np.float32)
    for row_offset in range(3):
        for column_offset in range(3):
            total += padded[
                row_offset : row_offset + values.shape[0],
                column_offset : column_offset + values.shape[1],
            ]
    return total / 9.0


def _gradient_discontinuity_score(rgb: np.ndarray) -> float:
    """Measure second derivatives after a 3x3 blur so fine dithering is ignored."""

    lab = rgb8_to_lab(np.asarray(rgb, dtype=np.uint8)).astype(np.float32)
    blurred = _box_blur(lab)
    horizontal = np.diff(blurred, n=2, axis=1)
    vertical = np.diff(blurred, n=2, axis=0)
    values = np.concatenate(
        (
            np.linalg.norm(horizontal, axis=2).ravel(),
            np.linalg.norm(vertical, axis=2).ravel(),
        )
    )
    return float(np.mean(values)) if values.size else 0.0


def _flat_neighbor_fraction(rgb: np.ndarray) -> float:
    """Measure palette plateaus that become visible as bands in smooth gradients."""

    values = np.asarray(rgb, dtype=np.uint8)
    horizontal = np.all(values[:, 1:] == values[:, :-1], axis=2)
    vertical = np.all(values[1:] == values[:-1], axis=2)
    equal_count = np.count_nonzero(horizontal) + np.count_nonzero(vertical)
    return float(equal_count / (horizontal.size + vertical.size))


@dataclass(frozen=True, slots=True)
class MappingQualityReport:
    pixel_count: int
    unique_input_colors: int
    unique_output_colors: int
    before_xy_out_of_gamut_fraction: float
    strict_invalid_pixel_count: int
    strict_invalid_pixel_fraction: float
    mean_delta_e76: float
    p95_delta_e76: float
    max_delta_e76: float
    mean_absolute_luminance_change: float
    abrupt_transition_fraction_before: float
    abrupt_transition_fraction_after: float
    gradient_discontinuity_before: float
    gradient_discontinuity_after: float
    flat_neighbor_fraction_before: float
    flat_neighbor_fraction_after: float
    control_abrupt_transition_fraction: float
    control_gradient_discontinuity: float
    control_flat_neighbor_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MappingQualityPolicy:
    """Delivery limits plus advisory limits for SDL gamut mapping."""

    maximum_out_of_gamut_fraction: float = 0.30
    maximum_preview_p95_delta_e76: float = 45.0
    maximum_preview_luminance_change: float = 0.02
    maximum_control_abrupt_transition_fraction: float = 0.04
    maximum_control_gradient_discontinuity: float = 1.5
    maximum_smooth_control_flat_neighbor_fraction: float = 0.30

    def failures(
        self,
        report: MappingQualityReport,
        *,
        method: str,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        checks = (
            (
                report.strict_invalid_pixel_count > 0,
                "strict_invalid_pixel_count",
                float(report.strict_invalid_pixel_count),
                0.0,
            ),
            (
                report.mean_absolute_luminance_change
                > self.maximum_preview_luminance_change,
                "mean_absolute_luminance_change",
                report.mean_absolute_luminance_change,
                self.maximum_preview_luminance_change,
            ),
            (
                report.control_abrupt_transition_fraction
                > self.maximum_control_abrupt_transition_fraction,
                "control_abrupt_transition_fraction",
                report.control_abrupt_transition_fraction,
                self.maximum_control_abrupt_transition_fraction,
            ),
            (
                report.control_gradient_discontinuity
                > self.maximum_control_gradient_discontinuity,
                "control_gradient_discontinuity",
                report.control_gradient_discontinuity,
                self.maximum_control_gradient_discontinuity,
            ),
        )
        for failed, name, value, limit in checks:
            if failed:
                failures.append(
                    f"{name}={value:.6f} violates its maximum {limit:.6f}"
                )
        if (
            method == "smooth"
            and report.control_flat_neighbor_fraction
            > self.maximum_smooth_control_flat_neighbor_fraction
        ):
            failures.append(
                "control_flat_neighbor_fraction="
                f"{report.control_flat_neighbor_fraction:.6f} violates its maximum "
                f"{self.maximum_smooth_control_flat_neighbor_fraction:.6f}"
            )
        return tuple(failures)

    def advisories(self, report: MappingQualityReport) -> tuple[str, ...]:
        """Report difficult source gamut without rejecting a safe mapped result."""

        advisories: list[str] = []
        if (
            report.before_xy_out_of_gamut_fraction
            > self.maximum_out_of_gamut_fraction
        ):
            advisories.append(
                "before_xy_out_of_gamut_fraction="
                f"{report.before_xy_out_of_gamut_fraction:.6f} exceeds its "
                f"advisory maximum {self.maximum_out_of_gamut_fraction:.6f}"
            )
        if report.p95_delta_e76 > self.maximum_preview_p95_delta_e76:
            advisories.append(
                f"p95_delta_e76={report.p95_delta_e76:.6f} exceeds its "
                f"advisory maximum {self.maximum_preview_p95_delta_e76:.6f}"
            )
        return tuple(advisories)


def analyze_mapping(
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    palette: SDLPalette,
    *,
    strict_rgb: np.ndarray | None = None,
) -> MappingQualityReport:
    """Measure preview fidelity and strict SDL control-map compliance."""

    source = np.asarray(input_rgb, dtype=np.uint8)
    mapped = np.asarray(output_rgb, dtype=np.uint8)
    if source.shape != mapped.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("input and output must be equal-sized RGB arrays")
    control = mapped if strict_rgb is None else np.asarray(strict_rgb, dtype=np.uint8)
    if control.shape != source.shape:
        raise ValueError("strict_rgb must have the same shape as input_rgb")

    inside_before = palette.chromaticity_is_inside(source)
    strict_valid = palette.strict_table_mask(control)
    source_lab = rgb8_to_lab(source)
    mapped_lab = rgb8_to_lab(mapped)
    delta_e = np.linalg.norm(source_lab - mapped_lab, axis=2)
    luminance_change = np.abs(relative_luminance(source) - relative_luminance(mapped))
    pixel_count = source.shape[0] * source.shape[1]
    invalid_count = int(np.count_nonzero(~strict_valid))
    return MappingQualityReport(
        pixel_count=pixel_count,
        unique_input_colors=len(np.unique(source.reshape(-1, 3), axis=0)),
        unique_output_colors=len(np.unique(mapped.reshape(-1, 3), axis=0)),
        before_xy_out_of_gamut_fraction=float((~inside_before).mean()),
        strict_invalid_pixel_count=invalid_count,
        strict_invalid_pixel_fraction=float(invalid_count / pixel_count),
        mean_delta_e76=float(delta_e.mean()),
        p95_delta_e76=float(np.percentile(delta_e, 95)),
        max_delta_e76=float(delta_e.max()),
        mean_absolute_luminance_change=float(luminance_change.mean()),
        abrupt_transition_fraction_before=_abrupt_transition_fraction(source),
        abrupt_transition_fraction_after=_abrupt_transition_fraction(mapped),
        gradient_discontinuity_before=_gradient_discontinuity_score(source),
        gradient_discontinuity_after=_gradient_discontinuity_score(mapped),
        flat_neighbor_fraction_before=_flat_neighbor_fraction(source),
        flat_neighbor_fraction_after=_flat_neighbor_fraction(mapped),
        control_abrupt_transition_fraction=_abrupt_transition_fraction(control),
        control_gradient_discontinuity=_gradient_discontinuity_score(control),
        control_flat_neighbor_fraction=_flat_neighbor_fraction(control),
    )
