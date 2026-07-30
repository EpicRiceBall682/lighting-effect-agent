"""Deterministic scene-to-pattern extraction for abstract luminaire graphics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ALLOWED_NAMED_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("pale golden yellow", (246, 218, 126)),
    ("pale yellow", (250, 231, 153)),
    ("pure yellow", (255, 224, 76)),
    ("light blue", (151, 203, 238)),
    ("sky blue", (145, 199, 235)),
    ("warm orange", (238, 154, 91)),
    ("soft orange", (242, 174, 112)),
    ("warm peach", (242, 183, 157)),
    ("light peach", (244, 197, 170)),
    ("pale pink", (242, 190, 207)),
    ("soft pink", (239, 177, 203)),
    ("light purple", (197, 180, 225)),
    ("pale lavender", (211, 199, 230)),
    ("lavender", (194, 174, 219)),
    ("ivory", (250, 241, 211)),
    ("coral", (237, 151, 139)),
    ("amber", (237, 178, 84)),
    ("red", (231, 115, 109)),
)

THEME_DEFAULT_PALETTES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "flowing": ((151, 203, 238), (242, 190, 207), (246, 218, 126)),
    "breathing": ((250, 231, 153), (242, 183, 157), (197, 180, 225)),
    "radiant": ((250, 241, 211), (246, 218, 126), (242, 190, 207)),
}

DENSITY_TO_PIXEL_LEVEL = {
    "lowest": "low",
    "low": "low",
    "middle": "medium",
    "high": "high",
}


@dataclass(frozen=True, slots=True)
class PatternAttributes:
    """Serializable controls used by the programmatic pattern renderer."""

    theme: str
    motif: str
    layout: str
    motion: str
    light_source: str
    pixel_level: str
    pattern_strength: float
    palette: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if not self.theme.strip() or not self.motif.strip():
            raise ValueError("theme and motif cannot be empty")
        if self.light_source not in {"surface", "linear", "point"}:
            raise ValueError("light_source must be surface, linear, or point")
        if self.pixel_level not in {"low", "medium", "high"}:
            raise ValueError("pixel_level must be low, medium, or high")
        if not 0.0 <= self.pattern_strength <= 0.18:
            raise ValueError("pattern_strength must be from 0 to 0.18")
        if not 2 <= len(self.palette) <= 4:
            raise ValueError("palette must contain from 2 to 4 colors")
        for color in self.palette:
            if len(color) != 3 or any(not 0 <= channel <= 255 for channel in color):
                raise ValueError("palette colors must be RGB byte triplets")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["palette"] = [list(color) for color in self.palette]
        return value


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def _theme_signature(scene: str) -> tuple[str, str, str, str, float]:
    lowered = scene.casefold()
    if _contains_any(
        lowered,
        (
            "水波",
            "海边",
            "海滩",
            "沙滩",
            "大海",
            "海洋",
            "流动",
            "动态",
            "霓虹",
            "能量",
            "品牌",
            "运动",
            "体育",
            "篮球",
            "足球",
            "比赛",
            "flow",
            "water",
            "sports",
        ),
    ):
        return "flowing", "flowing", "diagonal", "energetic", 0.11
    if _contains_any(
        lowered,
        (
            "入口",
            "橱窗",
            "日出",
            "唤醒",
            "晨",
            "阳光",
            "中心",
            "聚焦",
            "entrance",
            "sunrise",
            "window",
        ),
    ):
        return "radiant", "radiant", "centered", "gentle", 0.09
    return "breathing", "breathing", "horizontal", "calm", 0.07


def _palette_from_effect(effect: str, theme: str) -> tuple[tuple[int, int, int], ...]:
    lowered = effect.casefold()
    selected: list[tuple[int, int, int]] = []
    for name, rgb in ALLOWED_NAMED_COLORS:
        if name in lowered and rgb not in selected:
            selected.append(rgb)
    for rgb in THEME_DEFAULT_PALETTES[theme]:
        if rgb not in selected:
            selected.append(rgb)
    return tuple(selected[:4])


def extract_theme(
    scene: str,
    module_01_attributes: dict[str, Any] | None = None,
    *,
    pattern_strength: float | None = None,
    light_source: str = "surface",
) -> PatternAttributes:
    """Extract a safe, deterministic abstract pattern without another model call."""

    scene = str(scene).strip()
    if len(scene) < 2:
        raise ValueError("scene must contain at least two characters")
    attributes = module_01_attributes or {}
    theme, motif, layout, motion, default_strength = _theme_signature(scene)
    effect = str(attributes.get("effect", ""))
    density = str(attributes.get("density", "middle")).strip().lower()
    strength = default_strength if pattern_strength is None else float(pattern_strength)
    return PatternAttributes(
        theme=theme,
        motif=motif,
        layout=layout,
        motion=motion,
        light_source=light_source,
        pixel_level=DENSITY_TO_PIXEL_LEVEL.get(density, "medium"),
        pattern_strength=strength,
        palette=_palette_from_effect(effect, theme),
    )
