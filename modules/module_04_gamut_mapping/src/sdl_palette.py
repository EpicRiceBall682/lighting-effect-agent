"""Parse and validate the organizer-provided SDL xy/RGB color table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .color_spaces import rgb8_to_lab, rgb8_to_xyy


DEFAULT_SDL_PATH = (
    Path(__file__).resolve().parents[3] / "reference_data" / "颜色信息" / "SDL2_0.txt"
)
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SDL_LINE = re.compile(
    rf"^\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)\s*,\s*"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$"
)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise 2D convex hull using Andrew's algorithm."""

    unique = sorted({(float(point[0]), float(point[1])) for point in np.asarray(points)})
    if len(unique) < 3:
        raise ValueError("SDL xy coordinates need at least three unique points")

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    # The organizer table was generated with decimal interpolation, so points
    # that are mathematically collinear can differ at ~1e-17. Treat those
    # floating-point residues as collinear to recover the real outer boundary.
    collinear_tolerance = 1e-12
    for point in unique:
        while (
            len(lower) >= 2
            and cross(lower[-2], lower[-1], point) <= collinear_tolerance
        ):
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while (
            len(upper) >= 2
            and cross(upper[-2], upper[-1], point) <= collinear_tolerance
        ):
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def points_in_convex_polygon(
    points: np.ndarray,
    polygon: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Vectorized point-in-convex-polygon test including the boundary."""

    samples = np.asarray(points, dtype=np.float64)
    shape = samples.shape[:-1]
    flat = samples.reshape(-1, 2)
    hull = np.asarray(polygon, dtype=np.float64)
    if hull.ndim != 2 or hull.shape[0] < 3 or hull.shape[1] != 2:
        raise ValueError("polygon must contain at least three xy vertices")
    inside = np.ones(flat.shape[0], dtype=bool)
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge = end - start
        relative = flat - start
        cross = edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        inside &= cross >= -tolerance
    return inside.reshape(shape)


def _rgb_codes(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.int64)
    return (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]


@dataclass(frozen=True, slots=True)
class SDLPalette:
    """SDL samples plus a strict unique RGB palette and its CIE representations."""

    source_path: Path
    xy_samples: np.ndarray
    rgb_samples: np.ndarray
    hull_xy: np.ndarray
    rgb: np.ndarray
    lab: np.ndarray
    rgb_codes: np.ndarray

    @classmethod
    def from_file(cls, path: Path = DEFAULT_SDL_PATH) -> "SDLPalette":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"SDL color table does not exist: {source}")

        xy_rows: list[tuple[float, float]] = []
        rgb_rows: list[tuple[int, int, int]] = []
        malformed: list[int] = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            match = _SDL_LINE.fullmatch(stripped)
            if not match:
                malformed.append(line_number)
                continue
            x, y = float(match.group(1)), float(match.group(2))
            rgb = tuple(int(match.group(index)) for index in (3, 4, 5))
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and x + y <= 1.0 + 1e-9):
                raise ValueError(f"SDL line {line_number} contains invalid CIE xy coordinates")
            if any(channel < 0 or channel > 255 for channel in rgb):
                raise ValueError(f"SDL line {line_number} contains an invalid RGB channel")
            xy_rows.append((x, y))
            rgb_rows.append(rgb)

        if malformed:
            preview = ", ".join(str(number) for number in malformed[:8])
            raise ValueError(f"SDL table contains malformed non-empty lines: {preview}")
        if len(xy_rows) < 3:
            raise ValueError("SDL table contains fewer than three valid colors")

        xy_samples = np.asarray(xy_rows, dtype=np.float64)
        rgb_samples = np.asarray(rgb_rows, dtype=np.uint8)
        _, first_indices = np.unique(rgb_samples, axis=0, return_index=True)
        unique_rgb = rgb_samples[np.sort(first_indices)]
        if len(unique_rgb) < 2:
            raise ValueError("SDL table needs at least two unique RGB control colors")
        return cls(
            source_path=source,
            xy_samples=xy_samples,
            rgb_samples=rgb_samples,
            hull_xy=convex_hull(xy_samples),
            rgb=unique_rgb,
            lab=rgb8_to_lab(unique_rgb).astype(np.float32),
            rgb_codes=np.unique(_rgb_codes(unique_rgb)),
        )

    def chromaticity_is_inside(self, rgb: np.ndarray) -> np.ndarray:
        """Test standard-sRGB chromaticity against the SDL xy convex hull."""

        xy = rgb8_to_xyy(rgb)[..., :2]
        return points_in_convex_polygon(xy, self.hull_xy)

    def strict_table_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Return True where an RGB pixel exactly belongs to the SDL table."""

        return np.isin(_rgb_codes(rgb), self.rgb_codes)

    def assert_strict_table(self, rgb: np.ndarray) -> None:
        """Raise when any output pixel is not an exact SDL RGB entry."""

        valid = self.strict_table_mask(rgb)
        invalid_count = int(np.count_nonzero(~valid))
        if invalid_count:
            raise ValueError(
                f"mapped image contains {invalid_count} RGB pixels outside the SDL table"
            )
