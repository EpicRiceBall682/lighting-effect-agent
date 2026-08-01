"""High-level module-1 agent orchestration."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol, Sequence

from modules.color_vocabulary import matched_color_spans

from .client import DeepSeekClient
from .prompt_builder import build_system_prompt, build_user_prompt
from .schemas import LightingEffectAttributes, LightingEffectValidationError


SPACE_DENSITY_REFERENCES = (
    (1.38, "lowest"),
    (18.9, "low"),
    (31.8, "middle"),
    (75.0, "high"),
)
EFFECT_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
EFFECT_LENGTH_FILLERS = {
    1: "smoothly",
    2: "flowing smoothly",
    3: "with smooth diffusion",
    4: "with smooth luminous diffusion",
    5: "with smooth luminous diffusion throughout",
}
SCENE_COLOR_REQUIREMENTS = (
    (("蓝", "blue"), ("light blue", "sky blue")),
    (("紫", "purple", "lavender"), ("purple", "lavender")),
    (("粉", "pink"), ("pink",)),
    (("黄", "yellow", "golden"), ("yellow", "amber")),
    (("橙", "orange"), ("orange", "peach", "amber")),
    (("红", "red", "coral"), ("red", "coral", "pink")),
    (("白", "white"), ("ivory", "white")),
)
CLEAR_SKY_CUES = ("蓝天", "天空", "sky")
CLOUD_CUES = ("白云", "云朵", "云彩", "cloud")
WARM_SKY_CUES = ("日出", "日落", "黄昏", "夕阳", "朝霞", "晚霞", "sunrise", "sunset")


def _scene_color_validation_error(
    scene_description: str,
    effect: str,
) -> str | None:
    scene = scene_description.casefold()
    lowered = effect.casefold()
    for cues, allowed_terms in SCENE_COLOR_REQUIREMENTS:
        if any(cue in scene for cue in cues) and not any(
            term in lowered for term in allowed_terms
        ):
            return (
                "effect must preserve the user's requested color family using "
                + " or ".join(allowed_terms)
            )
    if (
        any(cue in scene for cue in CLEAR_SKY_CUES)
        and any(cue in scene for cue in CLOUD_CUES)
        and not any(cue in scene for cue in WARM_SKY_CUES)
        and re.search(r"\b(?:yellow|orange|amber|red|pink|peach)\b", lowered)
    ):
        return (
            "a clear blue-sky and white-cloud scene must remain light blue and ivory "
            "without adding a warm yellow, orange, red, or pink region"
        )
    return None


def _normalize_effect_for_uniqueness(effect: str) -> str:
    """Normalize an English effect caption for batch-level duplicate checks."""

    return " ".join(EFFECT_WORD_PATTERN.findall(str(effect).casefold()))


def _effect_design_signature(effect: str) -> tuple[str, ...]:
    """Return the ordered supported colors that determine the rendered gradient."""

    return tuple(name for _start, _end, name, _rgb in matched_color_spans(effect))


def _repair_near_boundary_effect_length(
    raw: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Repair only small 30–50 word boundary misses deterministically."""

    effect = str(raw.get("effect", "")).strip()
    matches = list(EFFECT_WORD_PATTERN.finditer(effect))
    word_count = len(matches)
    repaired = dict(raw)
    if 25 <= word_count < 30:
        suffix = EFFECT_LENGTH_FILLERS[30 - word_count]
        repaired["effect"] = effect.rstrip(" ,.;:") + ", " + suffix + "."
        return repaired
    if 50 < word_count <= 55:
        repaired["effect"] = effect[: matches[49].end()].rstrip(" ,.;:") + "."
        return repaired
    return None


def density_for_space_size(space_size_m2: float) -> str:
    """Map arbitrary positive areas to the nearest organizer density reference."""

    value = float(space_size_m2)
    if value <= 0:
        raise ValueError("space_size_m2 must be greater than zero")
    return min(
        SPACE_DENSITY_REFERENCES,
        key=lambda reference: (abs(reference[0] - value), reference[0]),
    )[1]


class JsonCompletionClient(Protocol):
    def complete_json(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]: ...


class LightingPromptAgent:
    """Translate a scene request into validated lighting-effect attributes."""

    def __init__(
        self,
        client: JsonCompletionClient | None = None,
        *,
        validation_retries: int = 2,
    ) -> None:
        self.client = client or DeepSeekClient()
        self.validation_retries = validation_retries

    def generate(
        self,
        scene_description: str,
        *,
        hardware_width_mm: float | None = None,
        hardware_height_mm: float | None = None,
        space_size_m2: float | None = None,
        forbidden_effects: Sequence[str] = (),
        forbidden_design_effects: Sequence[str] = (),
    ) -> LightingEffectAttributes:
        forbidden_normalized = {
            _normalize_effect_for_uniqueness(effect)
            for effect in forbidden_effects
            if str(effect).strip()
        }
        forbidden_designs = {
            _effect_design_signature(effect)
            for effect in forbidden_design_effects
            if _effect_design_signature(effect)
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_user_prompt(
                    scene_description,
                    hardware_width_mm=hardware_width_mm,
                    hardware_height_mm=hardware_height_mm,
                    space_size_m2=space_size_m2,
                ),
            },
        ]

        for attempt in range(self.validation_retries + 1):
            raw = self.client.complete_json(messages)
            try:
                attributes = LightingEffectAttributes.from_mapping(raw)
                scene_color_error = _scene_color_validation_error(
                    scene_description,
                    attributes.effect,
                )
                if scene_color_error:
                    raise LightingEffectValidationError(scene_color_error)
                if (
                    _normalize_effect_for_uniqueness(attributes.effect)
                    in forbidden_normalized
                ):
                    design_signature = _effect_design_signature(attributes.effect)
                    if design_signature:
                        forbidden_designs.add(design_signature)
                    raise LightingEffectValidationError(
                        "effect duplicates a prompt already used for another scene; "
                        "create a materially distinct palette or color proportion for "
                        "this scene instead of merely rephrasing the same design"
                    )
                if _effect_design_signature(attributes.effect) in forbidden_designs:
                    raise LightingEffectValidationError(
                        "effect reuses the same ordered color design as a rejected prompt; "
                        "change at least one supported color or swap the dominant and "
                        "secondary color roles"
                    )
                if space_size_m2 is not None:
                    expected_density = density_for_space_size(space_size_m2)
                    if attributes.density != expected_density:
                        raise LightingEffectValidationError(
                            f"density must be {expected_density} for "
                            f"a {space_size_m2:g} m² space"
                        )
                return attributes
            except LightingEffectValidationError as exc:
                if str(exc) == "effect must contain 30 to 50 English words":
                    repaired = _repair_near_boundary_effect_length(raw)
                    if repaired is not None:
                        try:
                            attributes = LightingEffectAttributes.from_mapping(repaired)
                            scene_color_error = _scene_color_validation_error(
                                scene_description,
                                attributes.effect,
                            )
                            if scene_color_error:
                                raise LightingEffectValidationError(
                                    scene_color_error
                                )
                            if (
                                _normalize_effect_for_uniqueness(attributes.effect)
                                in forbidden_normalized
                            ):
                                design_signature = _effect_design_signature(
                                    attributes.effect
                                )
                                if design_signature:
                                    forbidden_designs.add(design_signature)
                                raise LightingEffectValidationError(
                                    "effect duplicates a prompt already used for another "
                                    "scene; create a materially distinct palette or color "
                                    "proportion for this scene instead of merely rephrasing "
                                    "the same design"
                                )
                            if (
                                _effect_design_signature(attributes.effect)
                                in forbidden_designs
                            ):
                                raise LightingEffectValidationError(
                                    "effect reuses the same ordered color design as a "
                                    "rejected prompt; change at least one supported color "
                                    "or swap the dominant and secondary color roles"
                                )
                            if space_size_m2 is not None:
                                expected_density = density_for_space_size(space_size_m2)
                                if attributes.density != expected_density:
                                    raise LightingEffectValidationError(
                                        f"density must be {expected_density} for "
                                        f"a {space_size_m2:g} m² space"
                                    )
                            return attributes
                        except LightingEffectValidationError:
                            pass
                if attempt >= self.validation_retries:
                    raise
                messages.extend(
                    [
                        {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
                        {
                            "role": "user",
                            "content": (
                                "The JSON failed validation: "
                                f"{exc}. Return a corrected JSON object only, using the exact required fields."
                            ),
                        },
                    ]
                )

        raise AssertionError("unreachable")
