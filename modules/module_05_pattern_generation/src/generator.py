"""Persist module-five themed light textures and their reproducibility metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

from PIL import Image

from modules.module_03_image_generation.src.quality import analyze_image_quality

from .pattern_renderer import render_pattern
from .theme_extractor import PatternAttributes, extract_theme


@dataclass(frozen=True, slots=True)
class PatternQualityPolicy:
    maximum_mean_absolute_change: float = 0.025
    maximum_mean_luminance_change: float = 0.022
    maximum_isolated_chroma_increase: float = 0.0005
    maximum_broad_chroma_increase: float = 0.0008
    maximum_broad_chroma_fraction: float = 0.003
    maximum_fallback_attempts: int = 3


@dataclass(frozen=True, slots=True)
class PatternGenerationResult:
    image_path: Path
    manifest_path: Path
    attributes: PatternAttributes
    render_report: dict[str, Any]


class PatternGenerator:
    """Extract a scene theme and add an abstract, hardware-friendly motif."""

    def generate(
        self,
        base_image: Image.Image,
        *,
        scene: str,
        module_01_attributes: dict[str, Any] | None,
        seed: int,
        output_dir: Path,
        pattern_strength: float | None = None,
        enabled: bool = True,
        filename_stem: str = "themed_light_effect",
        quality_policy: PatternQualityPolicy | None = None,
    ) -> PatternGenerationResult:
        attributes = extract_theme(
            scene,
            module_01_attributes,
            pattern_strength=0.0 if not enabled else pattern_strength,
        )
        policy = quality_policy or PatternQualityPolicy()
        base_rgb = base_image.convert("RGB")
        baseline_quality = analyze_image_quality(base_rgb)
        requested_strength = attributes.pattern_strength
        attempts: list[dict[str, Any]] = []
        themed = base_rgb.copy()
        render_report: dict[str, Any] = {}
        accepted = False
        fallback_reason = ""
        for attempt in range(policy.maximum_fallback_attempts + 1):
            strength = (
                0.0
                if not enabled
                else requested_strength * (0.5**attempt)
            )
            attempt_attributes = replace(attributes, pattern_strength=strength)
            candidate, candidate_report = render_pattern(
                base_rgb,
                attempt_attributes,
                seed=seed,
            )
            candidate_quality = analyze_image_quality(candidate)
            failures: list[str] = []
            if (
                float(candidate_report["mean_absolute_change"])
                > policy.maximum_mean_absolute_change
            ):
                failures.append("mean_absolute_change")
            if (
                float(candidate_report["mean_luminance_change"])
                > policy.maximum_mean_luminance_change
            ):
                failures.append("mean_luminance_change")
            if candidate_quality.isolated_chroma_fraction > (
                baseline_quality.isolated_chroma_fraction
                + policy.maximum_isolated_chroma_increase
            ):
                failures.append("isolated_chroma_increase")
            if candidate_quality.broad_chroma_fraction > max(
                policy.maximum_broad_chroma_fraction,
                baseline_quality.broad_chroma_fraction
                + policy.maximum_broad_chroma_increase,
            ):
                failures.append("broad_chroma_increase")
            attempts.append(
                {
                    "attempt": attempt,
                    "strength": strength,
                    "quality": candidate_quality.to_dict(),
                    "render": candidate_report,
                    "failures": failures,
                }
            )
            if not failures:
                themed = candidate
                render_report = dict(candidate_report)
                attributes = attempt_attributes
                accepted = True
                break
            fallback_reason = ", ".join(failures)

        if not accepted:
            attributes = replace(attributes, pattern_strength=0.0)
            themed, render_report = render_pattern(
                base_rgb,
                attributes,
                seed=seed,
            )
        render_report.update(
            {
                "requested_strength": requested_strength,
                "effective_strength": attributes.pattern_strength,
                "quality_status": (
                    "accepted"
                    if attributes.pattern_strength > 0.0
                    else "bypassed"
                ),
                "fallback_reason": fallback_reason,
                "attempts": attempts,
                "quality_policy": asdict(policy),
            }
        )
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{filename_stem}_seed_{seed}.png"
        manifest_path = output_dir / "module_05_pattern.json"
        themed.save(image_path, format="PNG")
        manifest = {
            "scene": str(scene).strip(),
            "enabled": bool(enabled),
            "attributes": attributes.to_dict(),
            "render": render_report,
            "image_path": str(image_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return PatternGenerationResult(
            image_path=image_path,
            manifest_path=manifest_path,
            attributes=attributes,
            render_report=render_report,
        )
