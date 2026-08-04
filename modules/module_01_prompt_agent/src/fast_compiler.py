"""Deterministic scene compiler for the latency-critical web path."""

from __future__ import annotations

from dataclasses import dataclass

from .agent import density_for_space_size
from .schemas import LightingEffectAttributes


COLOR_CUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("粉", "pink"), "soft pink"),
    (("红", "red"), "warm red"),
    (("橙", "orange"), "bright orange"),
    (("黄", "金", "yellow", "gold"), "bright yellow"),
    (("绿", "草", "森林", "green"), "bright green"),
    (("青色", "cyan"), "bright cyan"),
    (("蓝", "blue"), "bright blue"),
    (("紫", "purple", "lavender"), "vivid purple"),
    (("白", "white"), "soft white"),
    (("靛", "indigo"), "indigo"),
)

SUBJECT_CUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("花海", "花园", "鲜花", "草地", "草原"), "open meadow filled with flowers and grasses"),
    (("森林", "树林", "绿植", "植被"), "lush forest clearing with layered plants and natural textures"),
    (("咖啡", "coffee", "cafe"), "welcoming contemporary coffee shop with tables, chairs, and people"),
    (("酒店", "大堂", "hotel", "lobby"), "refined hotel interior with seating, architectural details, and guests"),
    (("卧室", "客房", "bedroom"), "comfortable bedroom with a bed, textiles, and calm interior details"),
    (("客厅", "living"), "modern living room with furniture, plants, and layered materials"),
    (("餐厅", "用餐", "restaurant"), "atmospheric restaurant with dining tables, guests, and refined materials"),
    (("海边", "沙滩", "海滩", "海洋", "seaside"), "wide seaside setting with water, sky, shoreline, and people"),
    (("办公室", "工作", "office"), "contemporary workspace with desks, people, and practical details"),
    (("运动", "体育", "健身", "球场", "gym", "sports"), "active sports space with athletes and recognizable equipment"),
    (("商店", "零售", "橱窗", "店铺", "retail"), "stylish retail interior with products, displays, and visitors"),
)


@dataclass(frozen=True, slots=True)
class FastPromptCompiler:
    """Produce shared concept and gradient controls without a network request."""

    def generate(
        self,
        scene_description: str,
        *,
        hardware_width_mm: float | None = None,
        hardware_height_mm: float | None = None,
        space_size_m2: float | None = None,
        **_kwargs: object,
    ) -> LightingEffectAttributes:
        scene = " ".join(str(scene_description).strip().split())
        if len(scene) < 2:
            raise ValueError("scene_description must contain at least two characters")

        colors = self._ordered_colors(scene)
        subject = self._subject(scene)
        mood, intensities = self._mood(scene)
        density = (
            density_for_space_size(float(space_size_m2))
            if space_size_m2 is not None
            else "middle"
        )
        effect = self._effect(colors, mood)
        color_placement = self._color_placement(colors)
        concept_prompt = (
            f"Cinematic view of a {subject}, with ambient lighting arranged as "
            f"{color_placement}, expressing a {mood} mood, with tangible objects, "
            "realistic materials, natural depth, balanced composition, no text, logos, "
            "or watermark."
        )
        return LightingEffectAttributes.from_mapping(
            {
                "density": density,
                "m_intensity": intensities[0],
                "k_intensity": intensities[1],
                "a_intensity": intensities[2],
                "effect": effect,
                "concept_prompt": concept_prompt,
            }
        )

    @staticmethod
    def _ordered_colors(scene: str) -> tuple[str, ...]:
        lowered = scene.casefold()
        located: list[tuple[int, str]] = []
        for cues, color in COLOR_CUES:
            positions = [lowered.find(cue.casefold()) for cue in cues]
            positions = [position for position in positions if position >= 0]
            if positions:
                located.append((min(positions), color))
        ordered = []
        for _position, color in sorted(located):
            if color not in ordered:
                ordered.append(color)
        if not ordered:
            if any(cue in lowered for cue in ("清新", "自然", "fresh", "nature")):
                ordered = ["pale yellow", "bright green", "light blue"]
            elif any(cue in lowered for cue in ("浪漫", "romantic")):
                ordered = ["soft pink", "vivid purple"]
            elif any(cue in lowered for cue in ("活力", "运动", "energetic")):
                ordered = ["bright orange", "bright cyan"]
            else:
                ordered = ["light peach", "warm yellow"]
        elif len(ordered) == 1:
            companion = {
                "bright green": "light blue",
                "bright cyan": "soft pink",
                "bright blue": "soft white",
                "soft pink": "pale yellow",
                "warm red": "bright orange",
            }.get(ordered[0], "soft white")
            ordered.insert(0, companion)
        return tuple(ordered[:4])

    @staticmethod
    def _subject(scene: str) -> str:
        lowered = scene.casefold()
        for cues, subject in SUBJECT_CUES:
            if any(cue.casefold() in lowered for cue in cues):
                return subject
        return "contemporary real-world scene with people, furniture, plants, and clear spatial context"

    @staticmethod
    def _mood(scene: str) -> tuple[str, tuple[int, int, int]]:
        lowered = scene.casefold()
        if any(cue in lowered for cue in ("活力", "运动", "热烈", "energetic")):
            return "vivid energetic", (82, 90, 72)
        if any(cue in lowered for cue in ("浪漫", "温馨", "romantic", "cozy")):
            return "warm romantic", (68, 76, 64)
        if any(cue in lowered for cue in ("清新", "自然", "fresh", "nature")):
            return "fresh natural", (72, 78, 76)
        if any(cue in lowered for cue in ("安静", "放松", "睡眠", "calm", "relax")):
            return "calm relaxing", (56, 62, 58)
        return "balanced welcoming", (70, 76, 68)

    @staticmethod
    def _color_placement(colors: tuple[str, ...]) -> str:
        if len(colors) >= 4:
            return (
                f"{colors[0]} on the left, {colors[1]} near the left center, "
                f"{colors[2]} near the right center, and dominant {colors[3]} on the right"
            )
        if len(colors) >= 3:
            return (
                f"{colors[0]} on the left, {colors[1]} through the center, and dominant "
                f"{colors[2]} across the right"
            )
        return (
            f"{colors[0]} on the left and dominant {colors[1]} across the center and right"
        )

    @classmethod
    def _effect(cls, colors: tuple[str, ...], mood: str) -> str:
        placement = cls._color_placement(colors)
        return (
            f"Wide panoramic color field with {placement}, forming a smooth horizontal "
            f"gradient with uniform vertical color, {mood} illumination, visual coherence, "
            "and an uninterrupted luminous surface throughout."
        )
