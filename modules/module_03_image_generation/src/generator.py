"""Generate and save reproducible raw light-effect images."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from PIL import Image

from .config import GenerationConfig
from .model_loader import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_LORA_PATH,
    load_pipeline,
)
from .prompt_guidance import apply_prompt_color_guidance
from .quality import (
    ImageQualityPolicy,
    analyze_image_quality,
    suppress_broad_chroma_artifacts,
    suppress_isolated_chroma_artifacts,
)
from .structured_gradient import (
    DEFAULT_TEXTURE_STRENGTH,
    render_structured_gradient,
)


PROMPT_SUFFIX = "no objects"
QUALITY_RETRY_SEED_OFFSET = 130363

DENSITY_PROMPTS = {
    "lowest": "simple two-stop horizontal gradient",
    "low": "clean two-stop horizontal gradient",
    "middle": "balanced three-stop horizontal gradient",
    "high": "broad three-stop horizontal gradient",
}


def attribute_prompt_fragments(
    source_attributes: dict[str, Any] | None,
) -> list[str]:
    """Translate module-1 controls into concise diffusion prompt fragments."""

    if not source_attributes:
        return []
    fragments: list[str] = []
    density = str(source_attributes.get("density", "")).strip().lower()
    if density in DENSITY_PROMPTS:
        fragments.append(DENSITY_PROMPTS[density])

    try:
        main = int(source_attributes["m_intensity"])
        key = int(source_attributes["k_intensity"])
        ambient = int(source_attributes["a_intensity"])
    except (KeyError, TypeError, ValueError):
        return fragments

    overall = (main + key + ambient) / 3.0
    if overall >= 78:
        brightness = "high-key"
    elif overall >= 58:
        brightness = "balanced-bright"
    else:
        brightness = "gently lit"

    if key - main >= 12:
        contrast = "soft focal contrast"
    elif ambient >= 70:
        contrast = "even ambient fill"
    else:
        contrast = "low contrast"
    fragments.append(f"{brightness} lighting with {contrast}")
    return fragments


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, truncation=False, add_special_tokens=True)
    input_ids = (
        encoded["input_ids"]
        if isinstance(encoded, dict)
        else getattr(encoded, "input_ids")
    )
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def build_effective_prompt(
    prompt: str,
    source_attributes: dict[str, Any] | None = None,
    *,
    tokenizer: Any | None = None,
) -> tuple[str, list[str], int | None]:
    """Compose a prompt without letting control fragments trigger CLIP truncation."""

    primary = prompt.strip().rstrip("., ")
    if not primary:
        raise ValueError("prompt cannot be empty")

    max_tokens = None
    if tokenizer is not None:
        configured_limit = int(getattr(tokenizer, "model_max_length", 77))
        # Some tokenizers expose a huge sentinel instead of a real model limit.
        max_tokens = min(configured_limit, 77)

    selected_controls: list[str] = []
    required_prompt = ", ".join((primary, PROMPT_SUFFIX))
    if max_tokens is not None and _token_count(tokenizer, required_prompt) > max_tokens:
        raise ValueError(
            "module-1 effect is too long for the SD 1.5 text encoder; "
            "shorten it while preserving color placement and gradient direction"
        )

    for fragment in attribute_prompt_fragments(source_attributes):
        candidate = ", ".join(
            (primary, *selected_controls, fragment, PROMPT_SUFFIX)
        )
        if max_tokens is None or _token_count(tokenizer, candidate) <= max_tokens:
            selected_controls.append(fragment)

    effective_prompt = ", ".join(
        (primary, *selected_controls, PROMPT_SUFFIX)
    )
    token_count = (
        _token_count(tokenizer, effective_prompt) if tokenizer is not None else None
    )
    return effective_prompt, selected_controls, token_count


def enrich_prompt(
    prompt: str,
    source_attributes: dict[str, Any] | None = None,
) -> str:
    effective_prompt, _controls, _token_count_value = build_effective_prompt(
        prompt,
        source_attributes=source_attributes,
    )
    return effective_prompt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GenerationResult:
    image_path: Path
    manifest_path: Path
    seed: int
    width: int
    height: int
    diffusion_raw_path: Path | None = None
    guided_image_path: Path | None = None
    quality_retry_count: int = 0


@dataclass(frozen=True, slots=True)
class ConceptGenerationResult:
    image_path: Path
    manifest_path: Path
    seed: int
    width: int
    height: int
    steps: int
    inference_seconds: float
    acceleration: str


class LightEffectGenerator:
    """Keep one pipeline in memory and generate one image at a time."""

    def __init__(
        self,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        base_model_revision: str = DEFAULT_BASE_MODEL_REVISION,
        lora_path: Path = DEFAULT_LORA_PATH,
        device: str = "auto",
        lora_scale: float = 1.0,
        pipeline: Any | None = None,
        selected_device: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.base_model_revision = base_model_revision
        self.lora_path = Path(lora_path).expanduser().resolve()
        self.lora_scale = float(lora_scale)
        if pipeline is None:
            self.pipeline, self.device = load_pipeline(
                base_model=base_model,
                base_model_revision=base_model_revision,
                lora_path=self.lora_path,
                device=device,
                lora_scale=lora_scale,
            )
        else:
            self.pipeline = pipeline
            self.device = selected_device or device

    def generate_concept(
        self,
        prompt: str,
        *,
        output_dir: Path,
        seed: int,
        steps: int = 4,
        width: int = 512,
        height: int = 288,
    ) -> ConceptGenerationResult:
        """Generate one real-world concept image with a latency-oriented schedule."""

        import torch
        from diffusers import DPMSolverMultistepScheduler, LCMScheduler

        if not prompt.strip():
            raise ValueError("concept prompt cannot be empty")
        if width % 8 or height % 8 or min(width, height) < 64:
            raise ValueError("concept dimensions must be at least 64 and divisible by 8")
        steps = max(2, min(int(steps), 8))
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        original_scheduler = self.pipeline.scheduler
        lcm_available = bool(
            getattr(self.pipeline, "_concept_lcm_available", False)
        )
        acceleration = "lcm_lora_4_step" if lcm_available else "dpm_short_schedule"
        started = time.perf_counter()
        try:
            if lcm_available:
                self.pipeline.set_adapters("concept_lcm", adapter_weights=1.0)
                self.pipeline.scheduler = LCMScheduler.from_config(
                    original_scheduler.config
                )
                guidance_scale = 1.0
            else:
                if hasattr(self.pipeline, "disable_lora"):
                    self.pipeline.disable_lora()
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    original_scheduler.config,
                    algorithm_type="dpmsolver++",
                )
                guidance_scale = 4.5
                steps = max(6, steps)

            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            output = self.pipeline(
                prompt=prompt.strip(),
                negative_prompt=(
                    "text, letters, logo, watermark, deformed objects, duplicate people, "
                    "low detail, oversaturated, underexposed"
                ),
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            image = output.images[0].convert("RGB")
        finally:
            self.pipeline.scheduler = original_scheduler
            if hasattr(self.pipeline, "enable_lora"):
                self.pipeline.enable_lora()
            self.pipeline.set_adapters(
                "light_effect",
                adapter_weights=self.lora_scale,
            )

        elapsed = time.perf_counter() - started
        image_path = output_dir / f"concept_image_seed_{seed}.png"
        manifest_path = output_dir / "concept_image.json"
        image.save(image_path, format="PNG", optimize=False)
        manifest_path.write_text(
            json.dumps(
                {
                    "prompt": prompt.strip(),
                    "seed": int(seed),
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "device": self.device,
                    "acceleration": acceleration,
                    "inference_seconds": elapsed,
                    "image_path": str(image_path),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return ConceptGenerationResult(
            image_path=image_path,
            manifest_path=manifest_path,
            seed=int(seed),
            width=width,
            height=height,
            steps=steps,
            inference_seconds=elapsed,
            acceleration=acceleration,
        )

    def generate(
        self,
        prompt: str,
        *,
        output_dir: Path,
        config: GenerationConfig,
        enrich: bool = True,
        filename_stem: str = "raw_light_effect",
        source_attributes: dict[str, Any] | None = None,
        color_guidance_strength: float = 0.64,
        structured_gradient: bool = True,
        lora_texture_strength: float = DEFAULT_TEXTURE_STRENGTH,
        quality_policy: ImageQualityPolicy | None = None,
        max_quality_retries: int = 2,
    ) -> GenerationResult:
        import torch

        tokenizer = getattr(self.pipeline, "tokenizer", None)
        if enrich:
            effective_prompt, prompt_controls, prompt_token_count = build_effective_prompt(
                prompt,
                source_attributes=source_attributes,
                tokenizer=tokenizer,
            )
        else:
            effective_prompt = prompt.strip()
            prompt_controls = []
            prompt_token_count = (
                _token_count(tokenizer, effective_prompt)
                if tokenizer is not None
                else None
            )
        if not effective_prompt:
            raise ValueError("prompt cannot be empty")

        if max_quality_retries < 0:
            raise ValueError("max_quality_retries must be zero or greater")
        policy = quality_policy or ImageQualityPolicy()
        if structured_gradient and policy.minimum_mean_luminance > 0.22:
            # Several vivid organizer gradients intentionally sit near Y=0.24.
            # Rerolling diffusion cannot materially brighten a deterministic
            # palette and only changes the retained weak texture.
            policy = replace(policy, minimum_mean_luminance=0.22)
        quality_attempts: list[dict[str, Any]] = []
        accepted_seed = config.seed
        raw_image: Image.Image | None = None
        guided_image: Image.Image | None = None
        image: Image.Image | None = None
        color_guidance: dict[str, object] = {}
        artifact_cleanup: dict[str, float | bool] = {}
        quality = None

        for attempt in range(max_quality_retries + 1):
            accepted_seed = (
                config.seed + attempt * QUALITY_RETRY_SEED_OFFSET
            ) % (2**63)
            torch_generator = torch.Generator(device="cpu").manual_seed(accepted_seed)
            output: Any = self.pipeline(
                prompt=effective_prompt,
                negative_prompt=config.negative_prompt,
                width=config.width,
                height=config.height,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=torch_generator,
            )
            raw_image = output.images[0].convert("RGB")
            if structured_gradient:
                guided_image, color_guidance = render_structured_gradient(
                    raw_image,
                    prompt,
                    texture_strength=lora_texture_strength,
                )
            else:
                guided_image, color_guidance = apply_prompt_color_guidance(
                    raw_image,
                    prompt,
                    strength=color_guidance_strength,
                )
            image, artifact_cleanup = suppress_isolated_chroma_artifacts(
                guided_image
            )
            quality = analyze_image_quality(image)
            broad_cleanup: dict[str, float | bool] = {
                "applied": False,
                "detected_fraction": 0.0,
                "changed_pixel_fraction": 0.0,
                "remaining_broad_chroma_fraction": quality.broad_chroma_fraction,
            }
            if (
                quality.broad_chroma_fraction
                > policy.maximum_broad_chroma_fraction
            ):
                image, broad_cleanup = suppress_broad_chroma_artifacts(image)
                quality = analyze_image_quality(image)
            artifact_cleanup = {
                **artifact_cleanup,
                "broad_cleanup": broad_cleanup,
            }
            prompt_layout_error = color_guidance.get(
                "post_guidance_layout_error"
            )
            prompt_anchor_color_error = color_guidance.get(
                "post_guidance_anchor_color_error"
            )
            failures = policy.failures(
                quality,
                prompt_layout_error=(
                    float(prompt_layout_error)
                    if prompt_layout_error is not None
                    else None
                ),
                prompt_anchor_color_error=(
                    float(prompt_anchor_color_error)
                    if prompt_anchor_color_error is not None
                    else None
                ),
            )
            quality_attempts.append(
                {
                    "attempt": attempt,
                    "seed": accepted_seed,
                    "quality": quality.to_dict(),
                    "prompt_layout_error": prompt_layout_error,
                    "prompt_anchor_color_error": prompt_anchor_color_error,
                    "failures": list(failures),
                }
            )
            if not failures:
                break

        assert raw_image is not None
        assert guided_image is not None
        assert image is not None
        assert quality is not None
        final_layout_error = color_guidance.get("post_guidance_layout_error")
        final_anchor_color_error = color_guidance.get(
            "post_guidance_anchor_color_error"
        )
        final_failures = policy.failures(
            quality,
            prompt_layout_error=(
                float(final_layout_error)
                if final_layout_error is not None
                else None
            ),
            prompt_anchor_color_error=(
                float(final_anchor_color_error)
                if final_anchor_color_error is not None
                else None
            ),
        )
        if final_failures:
            attempt_summaries = " | ".join(
                "attempt "
                f"{attempt['attempt']} seed {attempt['seed']}: "
                + "; ".join(attempt["failures"])
                for attempt in quality_attempts
            )
            raise RuntimeError(
                "generated image failed quality policy after "
                f"{max_quality_retries + 1} attempts: "
                + attempt_summaries
            )

        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{filename_stem}_seed_{accepted_seed}.png"
        raw_path = (
            output_dir
            / f"{filename_stem}_seed_{accepted_seed}_diffusion_raw.png"
        )
        guided_path = (
            output_dir / f"{filename_stem}_seed_{accepted_seed}_guided.png"
        )
        manifest_path = output_dir / f"{filename_stem}_seed_{accepted_seed}.json"
        raw_image.save(raw_path, format="PNG")
        guided_image.save(guided_path, format="PNG")
        image.save(image_path, format="PNG")

        config_payload = config.to_dict()
        config_payload["seed"] = accepted_seed
        manifest = {
            "prompt": prompt.strip(),
            "effective_prompt": effective_prompt,
            "base_model": self.base_model,
            "base_model_revision": self.base_model_revision,
            "lora_path": str(self.lora_path),
            "lora_sha256": sha256_file(self.lora_path),
            "device": self.device,
            "module_01_attributes": source_attributes,
            "module_01_prompt_controls": prompt_controls,
            "effective_prompt_token_count": prompt_token_count,
            "prompt_color_guidance": color_guidance,
            "generation_mode": (
                "structured_gradient" if structured_gradient else "legacy_diffusion"
            ),
            "lora_texture_strength": (
                lora_texture_strength if structured_gradient else None
            ),
            "artifact_cleanup": artifact_cleanup,
            "quality_policy": asdict(policy),
            "quality_attempts": quality_attempts,
            "quality_status": "accepted",
            "quality_retry_count": len(quality_attempts) - 1,
            **config_payload,
            "quality": quality.to_dict(),
            "image_path": str(image_path),
            "diffusion_raw_path": str(raw_path),
            "guided_image_path": str(guided_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return GenerationResult(
            image_path=image_path,
            manifest_path=manifest_path,
            seed=accepted_seed,
            width=config.width,
            height=config.height,
            diffusion_raw_path=raw_path,
            guided_image_path=guided_path,
            quality_retry_count=len(quality_attempts) - 1,
        )
