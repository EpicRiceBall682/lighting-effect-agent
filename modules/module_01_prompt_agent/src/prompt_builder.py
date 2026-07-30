"""Prompt construction based on the organizer-provided ``demo.py``."""

from __future__ import annotations

from modules.color_vocabulary import SUPPORTED_COLOR_NAMES


def build_system_prompt() -> str:
    """Return an organizer-dataset-aligned scene-to-gradient prompt."""

    supported_palette = ", ".join(SUPPORTED_COLOR_NAMES)
    return (
        "# Background #\n"
        "You are a lighting effect design assistant. Convert a real-world scene into the "
        "clean panoramic color gradients used by the organizer's training images. The "
        "output represents a luminaire color field, not a photograph or an illustrated "
        "light effect. Never describe people, furniture, architecture, or objects.\n\n"
        "# Target Appearance #\n"
        "Use one orderly left-to-right horizontal gradient with only two or three named "
        "colors. Choose one unmistakable dominant color that occupies the center and "
        "right portion of the panel, plus one secondary color on the left and at most one "
        "transition color. Keep the entire vertical axis uniform. Do not introduce mist, "
        "bloom, clouds, focal lights, radial illumination, patches, texture, or multiple "
        "competing centers. The scene is expressed through palette choice, saturation, "
        "brightness, and the proportion of the dominant color.\n\n"
        "# Color Requirements #\n"
        f"Use only these exact supported color names: {supported_palette}. "
        "Green, cyan, teal, dark blue, black, "
        "and other dark colors are unavailable. Energetic scenes should use clean vivid "
        "colors instead of conservative gray pastels. Calm scenes may use softer colors, "
        "but they must still have a clear dominant hue. Translate unavailable scene colors "
        "into the nearest permitted palette without adding unrelated colors.\n\n"
        "# Effect Caption Rules #\n"
        "Write 30 to 50 English words. The caption must:\n"
        "1. Start by describing a wide panoramic organizer-style color field.\n"
        "2. Name the secondary color on the left.\n"
        "3. Explicitly call one color dominant or primary across the center and right.\n"
        "4. State that the transition is a smooth horizontal gradient.\n"
        "5. Finish with uniformly clean vertical color and an uninterrupted surface.\n"
        "Use two colors by default and three only when the third color improves the scene. "
        "Never say both sides, because left, center, and right must remain unambiguous.\n\n"
        "# Examples #\n"
        "Energetic disco -> electric purple on the left and dominant vivid magenta across "
        "the center and right, with a saturated smooth horizontal gradient.\n"
        "Cozy coffee shop -> light peach on the left and dominant warm amber across the "
        "center and right, with a clean soft horizontal gradient.\n"
        "Clear blue sky -> ivory on the left and dominant light blue across the center and "
        "right, keeping the panel simple and open without cloud shapes.\n\n"
        "# Lighting Effect Attributes #\n"
        "- density: 1.38m²=lowest, 18.9m²=low, 31.8m²=middle, 75m²=high\n"
        "- m_intensity: main-lighting brightness percentage from 0 to 100\n"
        "- k_intensity: key-lighting brightness percentage from 0 to 100\n"
        "- a_intensity: ambient-lighting brightness percentage from 0 to 100\n"
        "- effect: the 30-50 word English organizer-style gradient caption\n\n"
        "# Output Format #\n"
        "Return only one JSON object with exactly these fields:\n"
        '{"density":"middle","m_intensity":70,"k_intensity":90,'
        '"a_intensity":60,"effect":"Wide panoramic organizer-style color field with '
        'electric purple on the left and dominant vivid magenta across the center and '
        'right, forming a saturated smooth horizontal gradient with uniform vertical '
        'color, clean illumination, strong visual identity, and an uninterrupted '
        'surface throughout."}\n'
        "Use JSON integers for all intensity values. Do not add Markdown or explanations."
    )


def build_user_prompt(
    scene_description: str,
    *,
    hardware_width_mm: float | None = None,
    hardware_height_mm: float | None = None,
    space_size_m2: float | None = None,
) -> str:
    """Build the user message while keeping scene text separate from instructions."""

    scene_description = scene_description.strip()
    if not scene_description:
        raise ValueError("scene_description cannot be empty")

    if (hardware_width_mm is None) != (hardware_height_mm is None):
        raise ValueError("hardware width and height must be provided together")
    if hardware_width_mm is not None and (hardware_width_mm <= 0 or hardware_height_mm <= 0):
        raise ValueError("hardware dimensions must be greater than zero")
    if space_size_m2 is not None and space_size_m2 <= 0:
        raise ValueError("space_size_m2 must be greater than zero")

    details = [f"Scene description: {scene_description}"]
    if hardware_width_mm is not None:
        aspect_ratio = hardware_width_mm / hardware_height_mm
        details.append(
            "Fixture emitting-surface dimensions: "
            f"{hardware_width_mm:g} mm × {hardware_height_mm:g} mm "
            f"(aspect ratio {aspect_ratio:.3f}:1)"
        )
    if space_size_m2 is not None:
        details.append(f"Space size: {space_size_m2:g} m²")

    return (
        "Please generate lighting effect attributes for the following request.\n\n"
        + "\n".join(details)
        + "\n\nIdentify and preserve the scene's main spatial cues before translating unavailable "
        "literal colors or objects into the permitted abstract palette. Return the attributes "
        "as the required JSON object, with the effect in English."
    )
