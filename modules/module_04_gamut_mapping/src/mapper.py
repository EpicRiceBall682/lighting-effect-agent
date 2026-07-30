"""Continuous visual gamut preview plus strict SDL hardware control mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from .color_spaces import (
    linear_to_srgb,
    relative_luminance,
    rgb8_to_lab,
    rgb8_to_xyy,
    srgb_to_linear,
    xyy_to_xyz,
    xyz_to_srgb,
)
from .metrics import MappingQualityPolicy, MappingQualityReport, analyze_mapping
from .sdl_palette import SDLPalette


_BAYER_8X8 = (
    np.array(
        [
            [0, 48, 12, 60, 3, 51, 15, 63],
            [32, 16, 44, 28, 35, 19, 47, 31],
            [8, 56, 4, 52, 11, 59, 7, 55],
            [40, 24, 36, 20, 43, 27, 39, 23],
            [2, 50, 14, 62, 1, 49, 13, 61],
            [34, 18, 46, 30, 33, 17, 45, 29],
            [10, 58, 6, 54, 9, 57, 5, 53],
            [42, 26, 38, 22, 41, 25, 37, 21],
        ],
        dtype=np.float32,
    )
    + 0.5
) / 64.0


@dataclass(frozen=True, slots=True)
class MappingResult:
    image: Image.Image
    control_image: Image.Image
    out_of_gamut_mask: Image.Image
    quality: MappingQualityReport
    quality_policy: MappingQualityPolicy
    quality_failures: tuple[str, ...]
    method: str

    @property
    def accepted(self) -> bool:
        return not self.quality_failures


class GamutMapper:
    """Create a smooth visual preview and a separate strict control map."""

    def __init__(self, palette: SDLPalette, *, batch_size: int = 4096) -> None:
        if batch_size < 64:
            raise ValueError("batch_size must be at least 64")
        self.palette = palette
        self.batch_size = batch_size
        self._palette_norm = np.sum(palette.lab * palette.lab, axis=1)

    def _nearest_two(self, target_lab: np.ndarray) -> tuple[np.ndarray, ...]:
        targets = np.asarray(target_lab, dtype=np.float32)
        first_indices = np.empty(len(targets), dtype=np.int32)
        second_indices = np.empty(len(targets), dtype=np.int32)
        first_distances = np.empty(len(targets), dtype=np.float32)
        second_distances = np.empty(len(targets), dtype=np.float32)
        palette_transpose = self.palette.lab.T

        for start in range(0, len(targets), self.batch_size):
            end = min(start + self.batch_size, len(targets))
            batch = targets[start:end]
            distances = (
                np.sum(batch * batch, axis=1)[:, None]
                + self._palette_norm[None, :]
                - 2.0 * np.matmul(batch, palette_transpose)
            )
            np.maximum(distances, 0.0, out=distances)
            pair = np.argpartition(distances, kth=1, axis=1)[:, :2]
            pair_distances = np.take_along_axis(distances, pair, axis=1)
            order = np.argsort(pair_distances, axis=1)
            sorted_pair = np.take_along_axis(pair, order, axis=1)
            sorted_distances = np.take_along_axis(pair_distances, order, axis=1)
            first_indices[start:end] = sorted_pair[:, 0]
            second_indices[start:end] = sorted_pair[:, 1]
            first_distances[start:end] = sorted_distances[:, 0]
            second_distances[start:end] = sorted_distances[:, 1]
        return first_indices, second_indices, first_distances, second_distances

    def _map_nearest(self, rgb: np.ndarray) -> np.ndarray:
        flat = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        unique, inverse = np.unique(flat, axis=0, return_inverse=True)
        first, _second, _first_distance, _second_distance = self._nearest_two(
            rgb8_to_lab(unique)
        )
        return self.palette.rgb[first][inverse].reshape(rgb.shape)

    def _map_ordered_dither(
        self,
        rgb: np.ndarray,
        *,
        dither_strength: float,
    ) -> np.ndarray:
        flat = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        unique, inverse = np.unique(flat, axis=0, return_inverse=True)
        first, second, first_distance, second_distance = self._nearest_two(rgb8_to_lab(unique))
        first_delta = np.sqrt(first_distance)
        second_delta = np.sqrt(second_distance)
        denominator = first_delta + second_delta
        second_probability = np.divide(
            first_delta,
            denominator,
            out=np.zeros_like(first_delta),
            where=denominator > 1e-8,
        )
        second_probability = np.clip(second_probability * dither_strength, 0.0, 0.5)

        first_pixels = self.palette.rgb[first][inverse].reshape(rgb.shape)
        second_pixels = self.palette.rgb[second][inverse].reshape(rgb.shape)
        probabilities = second_probability[inverse].reshape(rgb.shape[:2])
        height, width = probabilities.shape
        thresholds = np.tile(
            _BAYER_8X8,
            ((height + 7) // 8, (width + 7) // 8),
        )[:height, :width]
        choose_second = thresholds < probabilities
        return np.where(choose_second[..., None], second_pixels, first_pixels).astype(np.uint8)

    def map_rgb_array(
        self,
        rgb: np.ndarray,
        *,
        method: str = "smooth",
        dither_strength: float = 1.0,
        smooth_radius: float = 0.6,
    ) -> np.ndarray:
        """Map an RGB array to exact table colors using the selected method."""

        values = np.asarray(rgb)
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        if method not in {"nearest", "smooth"}:
            raise ValueError("method must be one of: nearest, smooth")
        if not 0.0 <= dither_strength <= 2.0:
            raise ValueError("dither_strength must be from 0 to 2")
        if smooth_radius < 0.0 or smooth_radius > 5.0:
            raise ValueError("smooth_radius must be from 0 to 5")
        rgb8 = np.clip(values, 0, 255).astype(np.uint8)

        if method == "nearest":
            mapped = self._map_nearest(rgb8)
        else:
            if smooth_radius:
                filtered = Image.fromarray(rgb8, mode="RGB").filter(
                    ImageFilter.GaussianBlur(radius=smooth_radius)
                )
                target = np.asarray(filtered, dtype=np.uint8)
            else:
                target = rgb8
            mapped = self._map_ordered_dither(target, dither_strength=dither_strength)
        self.palette.assert_strict_table(mapped)
        return mapped

    @staticmethod
    def brightness_preserving_preview(
        source_rgb: np.ndarray,
        control_rgb: np.ndarray,
    ) -> np.ndarray:
        """Scale full-power SDL control colors to the Raw image luminance.

        The organizer table records chromaticity controls near their maximum
        drive level. Uniformly scaling a control color in linear light keeps
        its color direction while restoring the Raw image's per-pixel Y.
        """

        source = np.asarray(source_rgb, dtype=np.uint8)
        control = np.asarray(control_rgb, dtype=np.uint8)
        source_y = relative_luminance(source)
        control_y = relative_luminance(control)
        scale = np.divide(
            source_y,
            control_y,
            out=np.zeros_like(source_y),
            where=control_y > 1e-12,
        )
        control_linear = srgb_to_linear(control.astype(np.float64) / 255.0)
        preview_linear = np.clip(control_linear * scale[..., None], 0.0, 1.0)
        return np.rint(linear_to_srgb(preview_linear) * 255.0).astype(np.uint8)

    @staticmethod
    def _project_xy_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Project 2D points to the nearest point on a convex polygon boundary."""

        samples = np.asarray(points, dtype=np.float64)
        shape = samples.shape
        flat = samples.reshape(-1, 2)
        hull = np.asarray(polygon, dtype=np.float64)
        best = np.empty_like(flat)
        best_distance = np.full(len(flat), np.inf, dtype=np.float64)
        for index, start in enumerate(hull):
            end = hull[(index + 1) % len(hull)]
            edge = end - start
            denominator = float(np.dot(edge, edge))
            relative = flat - start
            parameter = np.clip(
                np.sum(relative * edge, axis=1) / denominator,
                0.0,
                1.0,
            )
            candidate = start + parameter[:, None] * edge
            distance = np.sum((flat - candidate) ** 2, axis=1)
            replace = distance < best_distance
            best[replace] = candidate[replace]
            best_distance[replace] = distance[replace]
        return best.reshape(shape)

    def continuous_gamut_preview(
        self,
        source_rgb: np.ndarray,
        *,
        boundary_margin: float = 0.01,
    ) -> np.ndarray:
        """Compress only out-of-gamut chromaticities while preserving Raw pixels.

        This image is for human visual comparison, not direct hardware control.
        Pixels already inside the SDL xy boundary remain byte-for-byte unchanged.
        Out-of-gamut xy values are projected continuously to the nearest boundary,
        nudged slightly toward the palette centroid for numerical stability, and
        converted back while retaining their original luminance where possible.
        """

        if not 0.0 <= boundary_margin <= 0.05:
            raise ValueError("boundary_margin must be from 0 to 0.05")
        source = np.asarray(source_rgb, dtype=np.uint8)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("source_rgb must have shape (height, width, 3)")
        inside = self.palette.chromaticity_is_inside(source)
        if bool(np.all(inside)):
            return source.copy()

        source_xyy = rgb8_to_xyy(source)
        target_xyy = source_xyy.copy()
        outside_xy = source_xyy[..., :2][~inside]
        projected = self._project_xy_to_polygon(outside_xy, self.palette.hull_xy)
        if boundary_margin:
            centroid = np.mean(self.palette.hull_xy, axis=0)
            projected = (
                projected * (1.0 - boundary_margin)
                + centroid * boundary_margin
            )
        target_xyy[..., :2][~inside] = projected
        preview = np.rint(xyz_to_srgb(xyy_to_xyz(target_xyy)) * 255.0).astype(
            np.uint8
        )
        preview[inside] = source[inside]
        return preview

    def map_image(
        self,
        image: Image.Image,
        *,
        method: str = "smooth",
        dither_strength: float = 1.0,
        smooth_radius: float = 0.6,
        quality_policy: MappingQualityPolicy | None = None,
    ) -> MappingResult:
        """Map a PIL image, preserve alpha, and return compliance metrics."""

        rgba = image.convert("RGBA")
        rgba_array = np.asarray(rgba, dtype=np.uint8)
        source_rgb = rgba_array[:, :, :3]
        control_rgb = self.map_rgb_array(
            source_rgb,
            method=method,
            dither_strength=dither_strength,
            smooth_radius=smooth_radius,
        )
        preview_rgb = self.continuous_gamut_preview(source_rgb)
        if "A" in image.getbands():
            mapped_image = Image.fromarray(
                np.dstack((preview_rgb, rgba_array[:, :, 3])),
                mode="RGBA",
            )
            control_image = Image.fromarray(
                np.dstack((control_rgb, rgba_array[:, :, 3])),
                mode="RGBA",
            )
        else:
            mapped_image = Image.fromarray(preview_rgb, mode="RGB")
            control_image = Image.fromarray(control_rgb, mode="RGB")

        inside = self.palette.chromaticity_is_inside(source_rgb)
        mask = Image.fromarray(np.where(inside, 0, 255).astype(np.uint8), mode="L")
        quality = analyze_mapping(
            source_rgb,
            preview_rgb,
            self.palette,
            strict_rgb=control_rgb,
        )
        policy = quality_policy or MappingQualityPolicy()
        return MappingResult(
            image=mapped_image,
            control_image=control_image,
            out_of_gamut_mask=mask,
            quality=quality,
            quality_policy=policy,
            quality_failures=policy.failures(quality, method=method),
            method=method,
        )
