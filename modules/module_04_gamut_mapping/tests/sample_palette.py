"""Generated non-proprietary SDL-like palette used only by tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modules.module_04_gamut_mapping.src.color_spaces import rgb8_to_xyy


def write_sample_sdl_palette(path: Path) -> None:
    anchors = (
        np.array((255, 0, 0), dtype=np.float64),
        np.array((255, 255, 0), dtype=np.float64),
        np.array((0, 0, 255), dtype=np.float64),
    )
    colors: list[np.ndarray] = []
    for start, end in zip(anchors, anchors[1:] + anchors[:1]):
        for amount in np.linspace(0.0, 1.0, 129):
            colors.append(np.rint(start * (1.0 - amount) + end * amount))
    rgb = np.unique(np.asarray(colors, dtype=np.uint8), axis=0)
    xyy = rgb8_to_xyy(rgb)
    lines = [
        f"({color_xyy[0]:.12f},{color_xyy[1]:.12f}),"
        f"({int(color[0])},{int(color[1])},{int(color[2])})"
        for color_xyy, color in zip(xyy, rgb)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
