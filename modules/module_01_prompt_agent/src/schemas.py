"""Validated data structures for lighting-effect attributes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from modules.color_vocabulary import unsupported_color_terms


VALID_DENSITIES = frozenset({"lowest", "low", "middle", "high"})
FORBIDDEN_EFFECT_TERMS = (
    "black",
    "dark",
    "shadow",
    "green",
    "cyan",
    "teal",
    "turquoise",
    "mint",
    "lime",
    "olive",
    "emerald",
    "navy",
    "indigo",
    "黑色",
    "暗色",
    "阴影",
    "绿色",
    "青色",
    "墨绿",
    "深蓝",
)
SPATIAL_EFFECT_TERMS = (
    "upper",
    "above",
    "top",
    "lower",
    "below",
    "bottom",
    "left",
    "right",
    "center",
    "centre",
    "horizontal",
    "vertical",
    "diagonal",
    "across",
    "outward",
    "inward",
    "horizon",
)
VERTICAL_UPPER_TERMS = ("upper", "above", "top")
VERTICAL_LOWER_TERMS = ("lower", "below", "bottom")
HORIZONTAL_LEFT_TERMS = ("left",)
HORIZONTAL_RIGHT_TERMS = ("right",)
TEXTURE_EFFECT_TERMS = (
    "gradient",
    "glow",
    "mist",
    "misty",
    "diffused",
    "diffusion",
    "bloom",
    "luminous",
    "cloud-like",
    "cloudlike",
)
ARTIFACT_PRONE_EFFECT_TERMS = (
    "spot",
    "spots",
    "speckle",
    "speckles",
    "dot",
    "dots",
    "blob",
    "blobs",
    "cluster",
    "clusters",
    "accent",
    "accents",
    "streak",
    "streaks",
    "beam",
    "beams",
    "hard line",
    "hard lines",
)


class LightingEffectValidationError(ValueError):
    """Raised when a model response does not satisfy the module contract."""


def _percentage(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise LightingEffectValidationError(f"{field_name} must be an integer from 0 to 100")

    if isinstance(value, str):
        value = value.strip()

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise LightingEffectValidationError(
            f"{field_name} must be an integer from 0 to 100"
        ) from exc

    if not numeric.is_integer() or not 0 <= numeric <= 100:
        raise LightingEffectValidationError(f"{field_name} must be an integer from 0 to 100")
    return int(numeric)


@dataclass(frozen=True, slots=True)
class LightingEffectAttributes:
    """The exact JSON contract requested by the original ``demo.py`` prompt."""

    density: str
    m_intensity: int
    k_intensity: int
    a_intensity: int
    effect: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LightingEffectAttributes":
        if not isinstance(value, Mapping):
            raise LightingEffectValidationError("model output must be a JSON object")

        required = {"density", "m_intensity", "k_intensity", "a_intensity", "effect"}
        missing = sorted(required.difference(value))
        if missing:
            raise LightingEffectValidationError(f"missing required fields: {', '.join(missing)}")
        unexpected = sorted(set(value).difference(required))
        if unexpected:
            raise LightingEffectValidationError(
                f"unexpected fields are not allowed: {', '.join(unexpected)}"
            )

        density = str(value["density"]).strip().lower()
        if density not in VALID_DENSITIES:
            allowed = ", ".join(sorted(VALID_DENSITIES))
            raise LightingEffectValidationError(f"density must be one of: {allowed}")

        effect = str(value["effect"]).strip()
        english_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", effect)
        if len(english_words) < 30 or len(english_words) > 50:
            raise LightingEffectValidationError(
                "effect must contain 30 to 50 English words"
            )
        if re.search(r"[\u3400-\u9fff]", effect):
            raise LightingEffectValidationError("effect must be written in English")

        lowered = effect.casefold()
        forbidden = []
        for term in FORBIDDEN_EFFECT_TERMS:
            normalized = term.casefold()
            if normalized.isascii():
                if re.search(rf"\b{re.escape(normalized)}\b", lowered):
                    forbidden.append(term)
            elif normalized in lowered:
                forbidden.append(term)
        if forbidden:
            raise LightingEffectValidationError(
                f"effect contains forbidden color or dark-tone terms: {', '.join(forbidden)}"
            )
        artifact_terms = [
            term
            for term in ARTIFACT_PRONE_EFFECT_TERMS
            if re.search(rf"\b{re.escape(term)}\b", lowered)
        ]
        if artifact_terms:
            raise LightingEffectValidationError(
                "effect contains artifact-prone visual terms: "
                + ", ".join(artifact_terms)
            )
        if re.search(r"\bblue\b", lowered) and not re.search(
            r"\b(?:light|pale|soft|bright|sky) blue\b", lowered
        ):
            raise LightingEffectValidationError(
                "blue is allowed only when explicitly described as a light or bright blue"
            )
        unsupported_colors = unsupported_color_terms(effect)
        if unsupported_colors:
            raise LightingEffectValidationError(
                "effect contains colors unsupported by the renderer: "
                + ", ".join(unsupported_colors)
            )
        if not any(
            re.search(rf"\b{re.escape(term)}\b", lowered)
            for term in SPATIAL_EFFECT_TERMS
        ):
            raise LightingEffectValidationError(
                "effect must describe spatial color placement or gradient direction"
            )
        if not any(
            re.search(rf"\b{re.escape(term)}\b", lowered)
            for term in TEXTURE_EFFECT_TERMS
        ):
            raise LightingEffectValidationError(
                "effect must include a smooth gradient"
            )
        vertical_layout = all(
            any(re.search(rf"\b{term}\b", lowered) for term in terms)
            for terms in (VERTICAL_UPPER_TERMS, VERTICAL_LOWER_TERMS)
        )
        horizontal_layout = all(
            any(re.search(rf"\b{term}\b", lowered) for term in terms)
            for terms in (HORIZONTAL_LEFT_TERMS, HORIZONTAL_RIGHT_TERMS)
        )
        horizontal_direction = bool(
            re.search(
                r"\bhorizontal(?:ly)?\s+(?:gradient|transition|fade|flow|blend|band)",
                lowered,
            )
        )
        vertical_direction = bool(
            re.search(
                r"\bvertical(?:ly)?\s+(?:gradient|transition|fade|flow|blend|band)",
                lowered,
            )
        )
        diagonal_direction = bool(re.search(r"\bdiagonal(?:ly)?\b", lowered))
        if vertical_layout and horizontal_direction and not (
            vertical_direction or diagonal_direction
        ):
            raise LightingEffectValidationError(
                "upper/lower color placement requires a vertical or diagonal transition, "
                "not a horizontal one"
            )
        if horizontal_layout and vertical_direction and not (
            horizontal_direction or diagonal_direction
        ):
            raise LightingEffectValidationError(
                "left/right color placement requires a horizontal or diagonal transition, "
                "not a vertical one"
            )

        return cls(
            density=density,
            m_intensity=_percentage(value["m_intensity"], "m_intensity"),
            k_intensity=_percentage(value["k_intensity"], "k_intensity"),
            a_intensity=_percentage(value["a_intensity"], "a_intensity"),
            effect=effect,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
