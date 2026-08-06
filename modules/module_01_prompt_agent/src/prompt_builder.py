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
        "Use one orderly left-to-right horizontal gradient with one to four named "
        "colors. Decide the palette, saturation, and color relationships independently "
        "from the complete scene meaning. A single color may occupy the entire panel when "
        "that is the clearest interpretation; when several colors are useful, give them "
        "unambiguous horizontal positions and identify the dominant color. Keep the entire "
        "vertical axis uniform. Do not introduce mist, "
        "bloom, clouds, focal lights, radial illumination, patches, texture, or multiple "
        "competing centers. The scene is expressed through palette choice, saturation, "
        "brightness, and the proportion of the dominant color.\n\n"
        "# Color Requirements #\n"
        f"Use only these exact supported color names: {supported_palette}. "
        "All hue families are equally available, including green, cyan, teal, navy, and "
        "indigo. Do not apply fixed mappings from an emotion, style, place, or activity to "
        "a hue family, and do not favor warm, cool, pastel, or vivid colors by default. "
        "Use your lighting-design judgment for this specific request. Preserve colors that "
        "the user explicitly requests, but distinguish a requested lighting color from a "
        "colored object whose name merely appears in the scene.\n\n"
        "# Effect Caption Rules #\n"
        "The effect field is the final independent lighting-color design. Interpret the same "
        "user request as the concept image, but choose the light-field palette using lighting "
        "design judgment rather than copying furniture, walls, plants, or pixel-area ratios "
        "from the concept scene. The runtime compares both results and only uses concept-image "
        "colors as a fallback when their dominant hues are severely inconsistent.\n"
        "Write 30 to 50 English words. The caption must:\n"
        "1. Start by describing a wide panoramic organizer-style color field.\n"
        "2. For a single-color design, state that it is dominant across the entire panel.\n"
        "3. For a multi-color design, name each color's horizontal placement and the "
        "dominant or primary color.\n"
        "4. State that the transition is a smooth horizontal gradient.\n"
        "5. Finish with uniformly clean vertical color and an uninterrupted surface.\n"
        "Use one to four colors solely according to the scene rather than a fixed default. "
        "Never say both sides, because left, center, and right must remain unambiguous.\n\n"
        "# Concept Image Prompt #\n"
        "Also write concept_prompt as an 8 to 80 word English scene-image prompt. Unlike "
        "effect, it must retain the real place, objects, plants, people when relevant, and "
        "the requested atmosphere. Treat it as an independent scene-design chain. Use natural, "
        "physically plausible material and lighting "
        "colors for the scene. Do not copy the effect field's left-to-right panel layout, do "
        "not tint every object with one palette, and do not apply a uniform color wash. "
        "Preserve an explicitly requested environmental light color only where it would "
        "naturally appear. Do not mention text, logos, brands, or a luminaire panel.\n\n"
        "# Lighting Effect Attributes #\n"
        "- density: 1.38m²=lowest, 18.9m²=low, 31.8m²=middle, 75m²=high\n"
        "- m_intensity: main-lighting brightness percentage from 0 to 100\n"
        "- k_intensity: key-lighting brightness percentage from 0 to 100\n"
        "- a_intensity: ambient-lighting brightness percentage from 0 to 100\n"
        "- effect: the 30-50 word English organizer-style gradient caption\n\n"
        "# Output Format #\n"
        "Return only one JSON object with exactly these fields:\n"
        '{"density":"middle","m_intensity":70,"k_intensity":90,'
        '"a_intensity":60,"effect":"Write the required 30 to 50 word English gradient '
        'caption here.","concept_prompt":"Write the corresponding English scene-image '
        'prompt here."}\n'
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
        + "\n\nInterpret the complete request before choosing colors. Select the palette independently "
        "for this scene without applying a canned style-to-color mapping. Treat a color as "
        "mandatory only when the user is actually requesting that lighting color, not merely "
        "mentioning a colored object. Return the attributes as the required JSON object, with "
        "the effect in English."
    )
