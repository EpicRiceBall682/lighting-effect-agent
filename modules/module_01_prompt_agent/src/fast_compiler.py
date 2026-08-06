"""Deterministic scene compiler for the latency-critical web path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

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
    (("琥珀", "amber"), "amber"),
    (("品红", "洋红", "magenta"), "vivid magenta"),
    (("海军蓝", "navy"), "deep navy"),
)

EXPLICIT_CHINESE_COLOR_CUES = (
    "粉",
    "红",
    "橙",
    "黄",
    "金色",
    "绿",
    "青色",
    "蓝",
    "紫",
    "白",
    "靛",
    "琥珀",
    "品红",
    "洋红",
    "海军蓝",
)
EXPLICIT_ENGLISH_COLOR_PATTERN = re.compile(
    r"\b(?:pink|red|orange|yellow|gold|green|cyan|teal|turquoise|blue|purple|"
    r"lavender|white|indigo|magenta|navy|amber|peach|coral|ivory)\b",
    re.IGNORECASE,
)

_NON_COLOR_PHRASES = (
    "黄昏",
    "黄金时段",
    "golden hour",
)


def _color_detection_text(scene: str) -> str:
    """Remove phrases whose color characters describe time rather than hue."""

    normalized = str(scene).casefold()
    for phrase in _NON_COLOR_PHRASES:
        normalized = normalized.replace(phrase, "")
    return normalized


def has_explicit_color_cue(scene: str) -> bool:
    """Return whether the user stated a color rather than only a mood or style."""

    text = _color_detection_text(scene)
    return any(cue in text for cue in EXPLICIT_CHINESE_COLOR_CUES) or bool(
        EXPLICIT_ENGLISH_COLOR_PATTERN.search(text)
    )


@dataclass(frozen=True, slots=True)
class StyleSpec:
    cues: tuple[str, ...]
    colors: tuple[str, ...]
    mood: str
    intensities: tuple[int, int, int]
    default_subject: str
    composition: str


@dataclass(frozen=True, slots=True)
class SemanticPaletteSpec:
    cues: tuple[str, ...]
    colors: tuple[str, ...]


SEMANTIC_PALETTE_SPECS: tuple[SemanticPaletteSpec, ...] = (
    SemanticPaletteSpec(
        ("晨雾", "薄雾", "morning mist"),
        ("soft white", "pale blue", "pale cyan"),
    ),
    SemanticPaletteSpec(
        ("日出", "晨曦", "朝霞", "sunrise"),
        ("deep navy", "warm peach", "bright yellow"),
    ),
    SemanticPaletteSpec(
        ("黄昏", "夕阳", "日落", "晚霞", "sunset", "golden hour"),
        ("warm orange", "soft pink", "vivid purple"),
    ),
    SemanticPaletteSpec(
        ("水波", "泳池", "清凉", "波光", "water ripple", "pool"),
        ("deep navy", "bright cyan", "light blue"),
    ),
    SemanticPaletteSpec(
        ("霓虹", "neon"),
        ("electric purple", "vivid magenta", "bright cyan"),
    ),
    SemanticPaletteSpec(
        ("涂鸦", "graffiti"),
        ("bright blue", "vivid magenta", "bright yellow", "bright green"),
    ),
    SemanticPaletteSpec(
        ("火焰", "沸腾", "烟火气", "flame", "boiling"),
        ("warm red", "bright orange", "bright yellow"),
    ),
    SemanticPaletteSpec(
        ("微光", "隐约", "神秘", "low light", "mysterious"),
        ("deep navy", "vivid purple", "amber"),
    ),
    SemanticPaletteSpec(
        ("地中海", "mediterranean"),
        ("bright blue", "light cyan", "soft white", "warm yellow"),
    ),
    SemanticPaletteSpec(
        ("花瓣", "晨露", "香气", "petal", "dew"),
        ("soft pink", "pale blue", "light purple"),
    ),
    SemanticPaletteSpec(
        ("魔法镜", "magic mirror"),
        ("indigo", "vivid purple", "soft pink"),
    ),
    SemanticPaletteSpec(
        ("品牌", "图腾", "brand", "totem"),
        ("bright blue", "vivid magenta", "warm orange"),
    ),
    SemanticPaletteSpec(
        ("季节", "seasonal"),
        ("soft purple", "pale blue", "warm peach"),
    ),
    SemanticPaletteSpec(
        ("自然光", "真实肤色", "daylight", "skin tone"),
        ("soft white", "pale blue", "light peach"),
    ),
    SemanticPaletteSpec(
        ("新鲜", "食欲", "fresh food", "appetite"),
        ("bright green", "bright yellow", "bright orange"),
    ),
    SemanticPaletteSpec(
        ("动态", "互动", "节奏", "流动", "dynamic", "interactive", "rhythm"),
        ("bright cyan", "bright blue", "vivid magenta"),
    ),
)


SCENE_PALETTE_VARIANTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "hotel": (
        ("soft white", "pale blue", "light cyan"),
        ("light peach", "warm yellow", "soft pink"),
        ("pale cyan", "soft purple", "warm peach"),
    ),
    "restaurant_bar": (
        ("amber", "warm orange", "soft pink"),
        ("warm peach", "pale golden yellow", "light purple"),
        ("deep navy", "vivid purple", "amber"),
    ),
    "cafe": (
        ("warm peach", "amber", "pale golden yellow"),
        ("pale green", "light cyan", "soft yellow"),
        ("soft pink", "warm peach", "light purple"),
    ),
    "retail": (
        ("bright blue", "vivid magenta", "light cyan"),
        ("soft pink", "bright yellow", "pale blue"),
        ("bright cyan", "vivid purple", "warm orange"),
    ),
    "bedroom": (
        ("pale blue", "soft purple", "soft pink"),
        ("warm peach", "pale golden yellow", "soft white"),
    ),
    "sports": (
        ("bright cyan", "bright blue", "bright green"),
        ("bright orange", "bright yellow", "vivid magenta"),
    ),
}

DEFAULT_PALETTE_VARIANTS: tuple[tuple[str, ...], ...] = (
    ("pale blue", "light cyan", "soft white"),
    ("warm peach", "soft pink", "pale golden yellow"),
    ("pale green", "soft yellow", "light blue"),
    ("soft purple", "pale blue", "warm peach"),
    ("bright cyan", "bright blue", "vivid purple"),
    ("amber", "warm orange", "soft pink"),
)


STYLE_SPECS: tuple[StyleSpec, ...] = (
    StyleSpec(
        ("赛博朋克", "cyberpunk", "霓虹未来"),
        ("electric purple", "vivid magenta", "bright cyan", "bright blue"),
        "futuristic neon energetic",
        (84, 92, 76),
        "a neon-lit futuristic city street with reflective surfaces and layered architecture",
        "high-contrast cyberpunk atmosphere with luminous signs and technological details",
    ),
    StyleSpec(
        ("未来主义", "科技感", "科幻", "futuristic", "sci-fi"),
        ("bright cyan", "bright blue", "vivid purple"),
        "clean futuristic dynamic",
        (78, 88, 72),
        "a futuristic architectural environment with advanced materials and technology",
        "clean high-tech composition with precise luminous details",
    ),
    StyleSpec(
        ("极简", "简约", "minimal", "minimalist"),
        ("soft white", "pale blue", "light cyan"),
        "quiet minimal refined",
        (58, 66, 56),
        "a minimal contemporary space with simple geometry and restrained materials",
        "uncluttered composition with generous negative space",
    ),
    StyleSpec(
        ("复古", "怀旧", "中古", "retro", "vintage"),
        ("warm orange", "amber", "soft pink"),
        "nostalgic warm expressive",
        (68, 78, 64),
        "a nostalgic retro setting with period furniture and recognizable vintage details",
        "cinematic vintage composition with tactile materials",
    ),
    StyleSpec(
        ("禅意", "侘寂", "东方", "zen", "wabi-sabi"),
        ("soft white", "pale green", "amber"),
        "calm contemplative natural",
        (54, 62, 56),
        "a tranquil contemplative space with natural materials and restrained objects",
        "balanced composition with quiet handcrafted details",
    ),
    StyleSpec(
        ("奢华", "高级感", "豪华", "luxury", "luxurious"),
        ("deep navy", "amber", "pale golden yellow"),
        "dramatic luxurious elegant",
        (66, 82, 62),
        "an elegant luxury environment with refined materials and architectural details",
        "layered cinematic composition with polished surfaces",
    ),
    StyleSpec(
        ("工业风", "粗犷", "industrial style"),
        ("deep navy", "amber", "warm orange"),
        "bold industrial dramatic",
        (70, 84, 66),
        "an industrial-style environment with metal structures and exposed materials",
        "strong architectural composition with functional details",
    ),
    StyleSpec(
        ("童趣", "梦幻", "可爱", "playful", "whimsical", "cute"),
        ("soft pink", "bright yellow", "light cyan", "light purple"),
        "playful imaginative cheerful",
        (76, 82, 74),
        "an imaginative playful setting with friendly forms and recognizable objects",
        "cheerful composition with soft rounded details",
    ),
)

@dataclass(frozen=True, slots=True)
class SceneSpec:
    name: str
    cues: tuple[str, ...]
    subject: str
    composition: str


SCENE_SPECS: tuple[SceneSpec, ...] = (
    SceneSpec(
        "sky_clouds",
        ("蓝天白云", "蓝天", "白云", "天空", "云层", "云朵", "晴空", "sky", "cloud"),
        "a vast blue sky with layered white clouds seen from below",
        "sky fills the entire frame, with no ground, water, buildings, or interiors",
    ),
    SceneSpec(
        "sunrise_sunset",
        ("日出", "晨曦", "朝霞", "夕阳", "日落", "晚霞", "黄昏", "sunrise", "sunset"),
        "an expansive outdoor horizon beneath a dramatic sunrise or sunset sky",
        "an open-air landscape composition without interior architecture or furniture",
    ),
    SceneSpec(
        "weather",
        ("雨天", "下雨", "雨景", "雪天", "下雪", "雪景", "雾天", "雨后", "rain", "snow", "fog"),
        "a recognizable outdoor landscape shaped by rain, snow, or atmospheric weather",
        "a weather-dominant open-air composition without indoor rooms or ceilings",
    ),
    SceneSpec(
        "flower_meadow",
        ("花海", "鲜花", "草地", "草原", "牧场", "meadow", "grassland"),
        "an open meadow filled with flowers and grasses",
        "a broad outdoor landscape with a visible horizon and no interior furnishings",
    ),
    SceneSpec(
        "forest",
        ("森林", "树林", "林间", "竹林", "绿植", "植被", "forest", "woodland"),
        "a lush forest clearing with layered plants and natural textures",
        "an outdoor woodland composition without rooms, walls, or furniture",
    ),
    SceneSpec(
        "mountain",
        ("雪山", "山谷", "山脉", "高山", "峡谷", "悬崖", "mountain", "valley", "canyon"),
        "a dramatic mountain landscape with layered peaks, terrain, and atmospheric depth",
        "a wide outdoor composition without interior architecture or furniture",
    ),
    SceneSpec(
        "inland_water",
        ("湖边", "湖面", "湖泊", "河边", "河流", "瀑布", "溪流", "lake", "river", "waterfall"),
        "a natural waterside landscape with reflective water, shoreline, and distant terrain",
        "an open-air composition without indoor rooms, furniture, walls, or ceilings",
    ),
    SceneSpec(
        "seaside",
        ("海边", "沙滩", "海滩", "海洋", "海面", "seaside", "beach", "ocean"),
        "a wide seaside setting with water, sky, shoreline, and recognizable coastal details",
        "an open-air coastal composition without interior rooms or furniture",
    ),
    SceneSpec(
        "desert",
        ("沙漠", "沙丘", "戈壁", "荒漠", "desert", "dune"),
        "an expansive desert landscape with sculpted dunes and a distant horizon",
        "a wide outdoor composition without rooms, furniture, walls, or ceilings",
    ),
    SceneSpec(
        "park_garden",
        ("公园", "花园", "庭院", "植物园", "园林", "park", "garden", "courtyard"),
        "a landscaped park or garden with paths, plants, seating, and visitors",
        "a coherent outdoor public-space composition without enclosed rooms",
    ),
    SceneSpec(
        "urban_outdoor",
        ("城市", "街道", "街头", "广场", "天际线", "夜景", "步行街", "city", "street", "plaza", "skyline"),
        "a recognizable urban outdoor setting with streets, architecture, and people",
        "an exterior city composition without indoor rooms, furniture, or ceilings",
    ),
    SceneSpec(
        "cafe",
        ("咖啡", "咖啡馆", "coffee", "cafe"),
        "a welcoming contemporary coffee shop with tables, chairs, and people",
        "a coherent indoor hospitality composition with recognizable cafe details",
    ),
    SceneSpec(
        "restaurant_bar",
        ("餐厅", "用餐", "酒吧", "清吧", "restaurant", "dining", "bar", "lounge"),
        "an atmospheric restaurant or lounge with tables, guests, and refined materials",
        "a coherent indoor hospitality composition with recognizable dining details",
    ),
    SceneSpec(
        "retail",
        ("商店", "零售", "橱窗", "店铺", "商场", "超市", "购物中心", "retail", "store", "mall"),
        "a stylish retail interior with products, displays, circulation, and visitors",
        "a coherent commercial interior with recognizable merchandise and fixtures",
    ),
    SceneSpec(
        "hotel",
        ("酒店", "大堂", "前台", "hotel", "lobby", "reception"),
        "a refined hotel interior with seating, architectural details, and guests",
        "a coherent hospitality interior with a visible reception or lounge context",
    ),
    SceneSpec(
        "bedroom",
        ("卧室", "客房", "睡眠", "bedroom", "guest room"),
        "a comfortable bedroom with a bed, textiles, and calm interior details",
        "a coherent residential interior with recognizable bedroom furnishings",
    ),
    SceneSpec(
        "living_room",
        ("客厅", "起居室", "会客", "living room"),
        "a modern living room with furniture, plants, and layered materials",
        "a coherent residential interior with recognizable seating and circulation",
    ),
    SceneSpec(
        "kitchen",
        ("厨房", "烹饪", "料理", "kitchen", "cooking"),
        "a contemporary kitchen with counters, cabinetry, appliances, and people",
        "a coherent residential interior with recognizable food-preparation details",
    ),
    SceneSpec(
        "bath_spa",
        ("浴室", "卫生间", "洗手间", "水疗", "温泉", "bathroom", "spa"),
        "a clean bathroom or spa interior with stone, water, and wellness details",
        "a coherent indoor wellness composition with realistic fixtures and materials",
    ),
    SceneSpec(
        "office",
        ("办公室", "办公", "会议室", "工作", "office", "workspace", "meeting room"),
        "a contemporary workspace with desks, people, and practical details",
        "a coherent indoor workplace composition with recognizable work activities",
    ),
    SceneSpec(
        "education",
        ("教室", "学校", "课堂", "图书馆", "阅读室", "classroom", "school", "library"),
        "a recognizable learning space with students, desks, books, and educational details",
        "a coherent education interior matching the described activity",
    ),
    SceneSpec(
        "healthcare",
        ("医院", "诊所", "病房", "候诊", "康复", "hospital", "clinic", "ward"),
        "a clean healthcare environment with staff, patients, and recognizable medical details",
        "a coherent clinical interior without hospitality or retail furnishings",
    ),
    SceneSpec(
        "culture_exhibition",
        ("博物馆", "美术馆", "画廊", "展厅", "展览", "museum", "gallery", "exhibition"),
        "a spacious museum or gallery with exhibits, visitors, and architectural details",
        "a coherent cultural interior with clearly visible displays and circulation",
    ),
    SceneSpec(
        "performance",
        ("剧院", "舞台", "影院", "演出", "音乐会", "theater", "stage", "cinema", "concert"),
        "a performance venue with a stage, audience, seating, and recognizable equipment",
        "a coherent entertainment interior focused on the performance area",
    ),
    SceneSpec(
        "sports",
        ("运动", "体育", "健身", "球场", "泳池", "gym", "sports", "stadium", "pool"),
        "an active sports space with athletes and recognizable equipment",
        "a coherent athletic setting matching the described sport",
    ),
    SceneSpec(
        "transport",
        ("机场", "车站", "火车站", "地铁", "候机", "候车", "airport", "station", "subway"),
        "a modern transport hub with passengers, circulation, signage forms, and vehicles",
        "a coherent transit environment with recognizable platforms or waiting areas",
    ),
    SceneSpec(
        "industrial",
        ("工厂", "车间", "仓库", "厂房", "生产线", "factory", "workshop", "warehouse"),
        "an industrial workspace with machinery, structural details, and workers",
        "a coherent production or storage environment with realistic equipment",
    ),
    SceneSpec(
        "circulation",
        ("走廊", "过道", "楼梯", "电梯厅", "通道", "corridor", "hallway", "staircase"),
        "an architectural circulation space with corridors, stairs, doors, and people",
        "a coherent interior perspective emphasizing movement and spatial depth",
    ),
)

DEFAULT_SCENE_SPEC = SceneSpec(
    "general",
    (),
    "a recognizable real-world setting matching the described activity",
    "a context-faithful composition without unrelated furniture or architecture",
)

@dataclass(frozen=True, slots=True)
class FastPromptCompiler:
    """Produce independent concept and gradient prompts without a network request."""

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

        scene_spec = self._scene_spec(scene)
        style_spec = self._style_spec(scene)
        colors = self._ordered_colors(
            scene,
            style_spec=style_spec,
            scene_spec=scene_spec,
        )
        mood, intensities = self._mood(scene, style_spec=style_spec)
        subject = scene_spec.subject
        composition = scene_spec.composition
        if scene_spec.name == "general" and style_spec is not None:
            subject = style_spec.default_subject
            composition = style_spec.composition
        density = (
            density_for_space_size(float(space_size_m2))
            if space_size_m2 is not None
            else "middle"
        )
        effect = self._effect(colors, mood)
        concept_prompt = (
            f"Cinematic view of {subject}. {composition}. Natural, physically plausible "
            f"scene lighting with a {mood} atmosphere, realistic material colors, detail, "
            "and depth without a uniform color wash."
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
    def _ordered_colors(
        scene: str,
        *,
        style_spec: StyleSpec | None = None,
        scene_spec: SceneSpec | None = None,
    ) -> tuple[str, ...]:
        lowered = scene.casefold()
        color_text = _color_detection_text(scene)
        located: list[tuple[int, int, str]] = []
        for cues, color in COLOR_CUES:
            matches = [
                (color_text.find(cue.casefold()), len(cue))
                for cue in cues
                if color_text.find(cue.casefold()) >= 0
            ]
            if matches:
                position, length = min(matches, key=lambda item: item[0])
                located.append((position, position + length, color))
        selected_spans: list[tuple[int, int]] = []
        ordered = []
        for start, end, color in sorted(
            located,
            key=lambda item: (item[0], -(item[1] - item[0])),
        ):
            if any(
                start < used_end and end > used_start
                for used_start, used_end in selected_spans
            ):
                continue
            selected_spans.append((start, end))
            if color not in ordered:
                ordered.append(color)
        if not ordered:
            semantic = next(
                (
                    spec.colors
                    for spec in SEMANTIC_PALETTE_SPECS
                    if any(cue.casefold() in lowered for cue in spec.cues)
                ),
                None,
            )
            if semantic is not None:
                ordered = list(semantic)
            elif style_spec is not None:
                ordered = list(style_spec.colors)
            else:
                if any(
                    cue in lowered for cue in ("清新", "自然", "fresh", "nature")
                ):
                    ordered = ["pale yellow", "bright green", "light blue"]
                elif any(cue in lowered for cue in ("浪漫", "romantic")):
                    ordered = ["soft pink", "vivid purple"]
                elif any(cue in lowered for cue in ("活力", "运动", "energetic")):
                    ordered = ["bright orange", "bright cyan"]
                else:
                    variants = SCENE_PALETTE_VARIANTS.get(
                        scene_spec.name if scene_spec is not None else "",
                        DEFAULT_PALETTE_VARIANTS,
                    )
                    digest = hashlib.sha256(scene.encode("utf-8")).digest()
                    index = int.from_bytes(digest[:2], "big") % len(variants)
                    ordered = list(variants[index])
        return tuple(ordered[:4])

    @staticmethod
    def _scene_spec(scene: str) -> SceneSpec:
        lowered = scene.casefold()
        for spec in SCENE_SPECS:
            if any(cue.casefold() in lowered for cue in spec.cues):
                return spec
        return DEFAULT_SCENE_SPEC

    @staticmethod
    def _style_spec(scene: str) -> StyleSpec | None:
        lowered = scene.casefold()
        for spec in STYLE_SPECS:
            if any(cue.casefold() in lowered for cue in spec.cues):
                return spec
        return None

    @classmethod
    def _subject(cls, scene: str) -> str:
        """Compatibility helper for callers that only need the subject phrase."""

        return cls._scene_spec(scene).subject

    @staticmethod
    def _mood(
        scene: str,
        *,
        style_spec: StyleSpec | None = None,
    ) -> tuple[str, tuple[int, int, int]]:
        lowered = scene.casefold()
        if any(
            cue in lowered
            for cue in (
                "活力",
                "运动",
                "热烈",
                "霓虹",
                "动态",
                "互动",
                "节奏",
                "energetic",
                "neon",
                "dynamic",
            )
        ):
            return "vivid energetic", (82, 90, 72)
        if any(cue in lowered for cue in ("浪漫", "温馨", "romantic", "cozy")):
            return "warm romantic", (68, 76, 64)
        if any(cue in lowered for cue in ("清新", "自然", "fresh", "nature")):
            return "fresh natural", (72, 78, 76)
        if any(
            cue in lowered
            for cue in (
                "暗调",
                "微光",
                "隐约",
                "神秘",
                "私密",
                "安静",
                "放松",
                "睡眠",
                "助眠",
                "calm",
                "relax",
                "low light",
            )
        ):
            return "calm relaxing", (56, 62, 58)
        if any(cue in lowered for cue in ("黄昏", "夕阳", "日落", "琥珀", "sunset")):
            return "warm atmospheric", (64, 74, 60)
        if any(cue in lowered for cue in ("正午", "日间明亮", "自然光", "daylight")):
            return "bright natural", (78, 84, 76)
        if style_spec is not None:
            return style_spec.mood, style_spec.intensities
        return "balanced welcoming", (70, 76, 68)

    @staticmethod
    def _color_placement(colors: tuple[str, ...]) -> str:
        if len(colors) == 1:
            return f"dominant {colors[0]} across the entire panel"
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
