"""High-level module-1 agent orchestration."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol, Sequence

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


def _normalize_effect_for_uniqueness(effect: str) -> str:
    """Normalize an English effect caption for batch-level duplicate checks."""

    return " ".join(EFFECT_WORD_PATTERN.findall(str(effect).casefold()))


def _repair_near_boundary_effect_length(
    raw: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Repair only small 30–50 word boundary misses deterministically."""

    effect = str(raw.get("effect", "")).strip()
    matches = list(EFFECT_WORD_PATTERN.finditer(effect))
    word_count = len(matches)
    repaired = dict(raw)
    if 15 <= word_count < 20:
        suffix = EFFECT_LENGTH_FILLERS[20 - word_count]
        repaired["effect"] = effect.rstrip(" ,.;:") + ", " + suffix + "."
        return repaired
    if 60 < word_count <= 65:
        repaired["effect"] = effect[: matches[59].end()].rstrip(" ,.;:") + "."
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
        # Retained in the public API for compatibility. Palette-level duplicate
        # rejection is intentionally disabled: two scenes may legitimately lead
        # the model to the same colors.
        _ = forbidden_design_effects
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
                if (
                    _normalize_effect_for_uniqueness(attributes.effect)
                    in forbidden_normalized
                ):
                    raise LightingEffectValidationError(
                        "effect duplicates a prompt already used for another scene; "
                        "reconsider the scene independently and return a distinct caption; "
                        "keep the same palette if it remains your best judgment"
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
                if str(exc) == "effect must contain 20 to 60 English words":
                    repaired = _repair_near_boundary_effect_length(raw)
                    if repaired is not None:
                        try:
                            attributes = LightingEffectAttributes.from_mapping(repaired)
                            if (
                                _normalize_effect_for_uniqueness(attributes.effect)
                                in forbidden_normalized
                            ):
                                raise LightingEffectValidationError(
                                    "effect duplicates a prompt already used for another "
                                    "scene; reconsider it independently and return a distinct "
                                    "caption; keep the same palette if it remains your best "
                                    "judgment"
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
