"""One-click module-1 -> module-3 -> module-5 -> module-4 demo pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Iterator
from uuid import uuid4
import zipfile

import numpy as np
from PIL import Image

from modules.module_01_prompt_agent.src.agent import LightingPromptAgent
from modules.module_01_prompt_agent.src.schemas import LightingEffectAttributes
from modules.module_03_image_generation.src.config import (
    DEFAULT_NEGATIVE_PROMPT,
    GenerationConfig,
)
from modules.module_03_image_generation.src.generator import LightEffectGenerator
from modules.module_03_image_generation.src.image_geometry import dimensions_from_fixture
from modules.module_03_image_generation.src.model_loader import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_LORA_PATH,
)
from modules.module_04_gamut_mapping.src.mapper import GamutMapper
from modules.module_04_gamut_mapping.src.sdl_palette import (
    DEFAULT_SDL_PATH,
    SDLPalette,
)
from modules.module_05_pattern_generation.src.generator import PatternGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "demo"
GREEN_SCENE_CUES = (
    "绿色",
    "草原",
    "草地",
    "森林",
    "树林",
    "绿植",
    "植被",
    "green",
    "grass",
    "forest",
)
SIMILARITY_DIFFERENCE_THRESHOLD = 0.03
SIMILARITY_RETRY_SEED_OFFSET = 104729
SDL_RETRY_SEED_OFFSET = 32452843
MAX_SDL_RETRIES = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_macos_launchctl_api_key() -> bool:
    """Copy the user's launchctl DeepSeek key into this process when needed."""

    if os.getenv("DEEPSEEK_API_KEY") or sys.platform != "darwin":
        return bool(os.getenv("DEEPSEEK_API_KEY"))
    completed = subprocess.run(
        ["launchctl", "getenv", "DEEPSEEK_API_KEY"],
        capture_output=True,
        text=True,
        check=False,
    )
    token = completed.stdout.strip()
    if token:
        os.environ["DEEPSEEK_API_KEY"] = token
        return True
    return False


def scene_palette_notice(scene: str) -> str:
    """Explain organizer/hardware color translation without discarding semantics."""

    lowered = scene.casefold()
    if any(cue in lowered for cue in GREEN_SCENE_CUES):
        return (
            "输入包含绿色或植被意象。赛题配色和 SDL 硬件色域不支持纯绿色；"
            "系统会保留开阔、上下层次和自然氛围，改用浅蓝、象牙白、"
            "淡黄或暖粉表达，而不会把场景结构直接删除。"
        )
    return ""


def scene_aware_seed(scene: str, base_seed: int) -> int:
    """Return a stable seed that changes when the normalized scene changes."""

    normalized = " ".join(str(scene).casefold().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    scene_value = int.from_bytes(digest[:8], "big")
    return (int(base_seed) + scene_value) % (2**63)


def image_mean_absolute_difference(
    first: Image.Image,
    second: Image.Image,
) -> float:
    """Measure low-resolution perceptual color difference on a 0-1 scale."""

    size = (64, 24)
    first_array = np.asarray(
        first.convert("RGB").resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    second_array = np.asarray(
        second.convert("RGB").resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return float(np.mean(np.abs(first_array - second_array)) / 255.0)


@dataclass(frozen=True, slots=True)
class DemoPipelineResult:
    """All user-facing outputs and reproducibility artifacts from one run."""

    run_dir: Path
    prompt: str
    attributes: dict[str, Any]
    raw_image_path: Path
    themed_image_path: Path
    sdl_preview_path: Path
    sdl_control_path: Path
    out_of_gamut_mask_path: Path
    prompt_json_path: Path
    generation_manifest_path: Path
    pattern_manifest_path: Path
    report_path: Path
    archive_path: Path
    quality: dict[str, int | float]
    raw_quality: dict[str, Any]
    artifact_cleanup: dict[str, Any]
    color_guidance: dict[str, Any]
    pattern_report: dict[str, Any]
    palette_notice: str
    width: int
    height: int
    requested_seed: int
    effective_seed: int
    seed_mode: str
    quality_retry_count: int
    similarity_retry_count: int
    similarity_difference: float | None
    sdl_retry_count: int


class LightingDemoPipeline:
    """Connect prompt generation, LoRA inference, and SDL gamut mapping.

    The diffusion model and SDL mapper are created lazily and then reused.
    A lock serializes inference because one cached diffusion pipeline should
    not serve concurrent requests on the same MPS/CUDA device.
    """

    def __init__(
        self,
        *,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        device: str = "auto",
        base_model: str = DEFAULT_BASE_MODEL,
        lora_path: Path = DEFAULT_LORA_PATH,
        sdl_path: Path = DEFAULT_SDL_PATH,
        prompt_agent: Any | None = None,
        prompt_agent_factory: Callable[[], Any] = LightingPromptAgent,
        generator_factory: Callable[..., Any] = LightEffectGenerator,
        mapper_factory: Callable[[SDLPalette], Any] = GamutMapper,
        pattern_generator: Any | None = None,
        lora_scale: float = 0.8,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.device = device
        self.base_model = base_model
        self.lora_path = Path(lora_path).expanduser().resolve()
        self.sdl_path = Path(sdl_path).expanduser().resolve()
        self.prompt_agent = prompt_agent
        self.prompt_agent_factory = prompt_agent_factory
        self.generator_factory = generator_factory
        self.mapper_factory = mapper_factory
        self.pattern_generator = pattern_generator or PatternGenerator()
        self.lora_scale = lora_scale
        self._generator: Any | None = None
        self._mapper: Any | None = None
        self._run_lock = threading.Lock()
        self._previous_image: Image.Image | None = None
        self._previous_scene: str | None = None
        self._previous_prompt: str | None = None

    def _get_prompt_agent(self) -> Any:
        if self.prompt_agent is None:
            load_macos_launchctl_api_key()
            self.prompt_agent = self.prompt_agent_factory()
        return self.prompt_agent

    def _get_generator(self) -> Any:
        if self._generator is None:
            self._generator = self.generator_factory(
                base_model=self.base_model,
                lora_path=self.lora_path,
                device=self.device,
                lora_scale=self.lora_scale,
            )
        return self._generator

    def _get_mapper(self) -> Any:
        if self._mapper is None:
            palette = SDLPalette.from_file(self.sdl_path)
            self._mapper = self.mapper_factory(palette)
        return self._mapper

    @staticmethod
    def _validate_inputs(
        scene: str,
        width_mm: float,
        height_mm: float,
        space_size_m2: float | None,
        seed: int,
        steps: int,
    ) -> tuple[str, float, float, float | None, int, int]:
        scene = str(scene).strip()
        if len(scene) < 4:
            raise ValueError("请至少输入 4 个字符的场景描述。")
        width_mm = float(width_mm)
        height_mm = float(height_mm)
        if (
            not math.isfinite(width_mm)
            or not math.isfinite(height_mm)
            or width_mm <= 0
            or height_mm <= 0
        ):
            raise ValueError("灯具宽度和高度必须大于 0。")
        if space_size_m2 in (None, ""):
            normalized_space = None
        else:
            normalized_space = float(space_size_m2)
            if not math.isfinite(normalized_space) or normalized_space <= 0:
                raise ValueError("空间面积必须大于 0，或者留空。")
        seed = int(seed)
        steps = int(steps)
        if not 0 <= seed < 2**63:
            raise ValueError("随机种子超出有效范围。")
        if not 1 <= steps <= 150:
            raise ValueError("生成步数必须在 1 到 150 之间。")
        return scene, width_mm, height_mm, normalized_space, seed, steps

    def _new_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / f"{timestamp}_{uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    @contextmanager
    def _managed_run_dir(self) -> Iterator[Path]:
        """Remove incomplete run artifacts when any pipeline stage fails."""

        run_dir = self._new_run_dir()
        try:
            yield run_dir
        except BaseException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

    @staticmethod
    def _remove_obsolete_generation(
        obsolete_generation: Any,
        current_generation: Any,
    ) -> None:
        current_paths = {
            path
            for path in (
                current_generation.image_path,
                current_generation.manifest_path,
                current_generation.diffusion_raw_path,
                current_generation.guided_image_path,
            )
            if path is not None
        }
        for obsolete in (
            obsolete_generation.image_path,
            obsolete_generation.manifest_path,
            obsolete_generation.diffusion_raw_path,
            obsolete_generation.guided_image_path,
        ):
            if (
                obsolete is not None
                and obsolete not in current_paths
                and obsolete.exists()
            ):
                obsolete.unlink()

    @staticmethod
    def _notify(progress: Callable[[float, str], Any] | None, value: float, text: str) -> None:
        if progress is not None:
            progress(value, text)

    def run(
        self,
        scene: str,
        width_mm: float,
        height_mm: float,
        space_size_m2: float | None = None,
        seed: int = 20260724,
        steps: int = 30,
        *,
        fixed_seed: bool = False,
        prompt_override: str | None = None,
        attributes_override: dict[str, Any] | None = None,
        pattern_enabled: bool = True,
        pattern_strength: float | None = None,
        progress: Callable[[float, str], Any] | None = None,
    ) -> DemoPipelineResult:
        """Run modules 1, 3, and 4 and persist every intermediate artifact."""

        values = self._validate_inputs(
            scene, width_mm, height_mm, space_size_m2, seed, steps
        )
        scene, width_mm, height_mm, space_size_m2, requested_seed, steps = values
        # Validate the final diffusion dimensions before calling an external API
        # or creating any output directory.
        width, height = dimensions_from_fixture(width_mm, height_mm)
        seed_mode = "fixed" if fixed_seed else "scene_derived"
        if pattern_strength is not None and not 0.0 <= float(pattern_strength) <= 0.18:
            raise ValueError("模块五增强强度必须在 0 到 0.18 之间。")
        effective_seed = (
            requested_seed
            if fixed_seed
            else scene_aware_seed(scene, requested_seed)
        )
        with self._run_lock, self._managed_run_dir() as run_dir:
            palette_notice = scene_palette_notice(scene)
            if prompt_override is not None:
                self._notify(progress, 0.08, "模块一：正在校验编辑后的提示词")
                if not attributes_override:
                    raise ValueError("缺少模块一结构化参数，请先完成一次自动生成。")
                edited = dict(attributes_override)
                edited["effect"] = str(prompt_override).strip()
                attributes = LightingEffectAttributes.from_mapping(edited)
                prompt_source = "user_edited"
            else:
                self._notify(progress, 0.08, "模块一：正在理解中文场景")
                attributes = self._get_prompt_agent().generate(
                    scene,
                    hardware_width_mm=width_mm,
                    hardware_height_mm=height_mm,
                    space_size_m2=space_size_m2,
                )
                prompt_source = "deepseek"
            attributes_dict = attributes.to_dict()
            prompt_json_path = run_dir / "module_01_prompt.json"
            prompt_json_path.write_text(
                json.dumps(attributes_dict, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            config = GenerationConfig(
                width=width,
                height=height,
                seed=effective_seed,
                num_inference_steps=steps,
                guidance_scale=7.0,
                lora_scale=self.lora_scale,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            )

            self._notify(
                progress,
                0.30,
                "模块三：正在生成主渐变并提取弱 LoRA 亮度纹理",
            )
            generation = self._get_generator().generate(
                attributes.effect,
                output_dir=run_dir,
                config=config,
                source_attributes=attributes_dict,
            )
            effective_seed = generation.seed

            with Image.open(generation.image_path) as opened:
                current_image = opened.convert("RGB").copy()

            scene_or_prompt_changed = (
                self._previous_image is not None
                and (
                    scene != self._previous_scene
                    or attributes.effect != self._previous_prompt
                )
            )
            initial_difference: float | None = None
            final_difference: float | None = None
            retry_count = 0
            if scene_or_prompt_changed and self._previous_image is not None:
                initial_difference = image_mean_absolute_difference(
                    self._previous_image,
                    current_image,
                )
                final_difference = initial_difference
                if (
                    initial_difference < SIMILARITY_DIFFERENCE_THRESHOLD
                    and not fixed_seed
                ):
                    retry_count = 1
                    retry_seed = (
                        effective_seed + SIMILARITY_RETRY_SEED_OFFSET
                    ) % (2**63)
                    retry_config = GenerationConfig(
                        width=width,
                        height=height,
                        seed=retry_seed,
                        num_inference_steps=steps,
                        guidance_scale=7.0,
                        lora_scale=self.lora_scale,
                        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    )
                    self._notify(
                        progress,
                        0.68,
                        "模块三：检测到结果过于相似，正在更换种子重试",
                    )
                    first_generation = generation
                    generation = self._get_generator().generate(
                        attributes.effect,
                        output_dir=run_dir,
                        config=retry_config,
                        source_attributes=attributes_dict,
                    )
                    with Image.open(generation.image_path) as opened:
                        current_image = opened.convert("RGB").copy()
                    effective_seed = generation.seed
                    final_difference = image_mean_absolute_difference(
                        self._previous_image,
                        current_image,
                    )
                    self._remove_obsolete_generation(
                        first_generation,
                        generation,
                    )

            generation_details = json.loads(
                generation.manifest_path.read_text(encoding="utf-8")
            )
            raw_quality = dict(generation_details.get("quality", {}))
            artifact_cleanup = dict(
                generation_details.get("artifact_cleanup", {})
            )
            color_guidance = dict(
                generation_details.get("prompt_color_guidance", {})
            )

            self._notify(progress, 0.73, "模块五：正在进行低频主题光场增强")
            pattern = self.pattern_generator.generate(
                current_image,
                scene=scene,
                module_01_attributes=attributes_dict,
                seed=effective_seed,
                output_dir=run_dir,
                pattern_strength=pattern_strength,
                enabled=pattern_enabled,
            )
            with Image.open(pattern.image_path) as opened:
                mapping_image = opened.convert("RGB").copy()

            self._notify(progress, 0.78, "模块四：正在进行 SDL 色域映射")
            mapped = self._get_mapper().map_image(mapping_image, method="smooth")
            sdl_attempts = [
                {
                    "attempt": 0,
                    "seed": effective_seed,
                    "quality": mapped.quality.to_dict(),
                    "failures": list(mapped.quality_failures),
                    "advisories": list(
                        mapped.quality_policy.advisories(mapped.quality)
                    ),
                }
            ]
            sdl_retry_count = 0
            if mapped.quality_failures and MAX_SDL_RETRIES and not fixed_seed:
                first_generation = generation
                first_pattern = pattern
                sdl_retry_count = 1
                retry_seed = (effective_seed + SDL_RETRY_SEED_OFFSET) % (2**63)
                retry_config = GenerationConfig(
                    width=width,
                    height=height,
                    seed=retry_seed,
                    num_inference_steps=steps,
                    guidance_scale=7.0,
                    lora_scale=self.lora_scale,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                )
                self._notify(
                    progress,
                    0.82,
                    "模块四：色域质量未通过，正在更换种子自动重试",
                )
                generation = self._get_generator().generate(
                    attributes.effect,
                    output_dir=run_dir,
                    config=retry_config,
                    source_attributes=attributes_dict,
                )
                effective_seed = generation.seed
                with Image.open(generation.image_path) as opened:
                    current_image = opened.convert("RGB").copy()
                generation_details = json.loads(
                    generation.manifest_path.read_text(encoding="utf-8")
                )
                raw_quality = dict(generation_details.get("quality", {}))
                artifact_cleanup = dict(
                    generation_details.get("artifact_cleanup", {})
                )
                color_guidance = dict(
                    generation_details.get("prompt_color_guidance", {})
                )
                pattern = self.pattern_generator.generate(
                    current_image,
                    scene=scene,
                    module_01_attributes=attributes_dict,
                    seed=effective_seed,
                    output_dir=run_dir,
                    pattern_strength=pattern_strength,
                    enabled=pattern_enabled,
                )
                with Image.open(pattern.image_path) as opened:
                    mapping_image = opened.convert("RGB").copy()
                mapped = self._get_mapper().map_image(
                    mapping_image,
                    method="smooth",
                )
                self._remove_obsolete_generation(first_generation, generation)
                if (
                    first_pattern.image_path != pattern.image_path
                    and first_pattern.image_path.exists()
                ):
                    first_pattern.image_path.unlink()
                sdl_attempts.append(
                    {
                        "attempt": 1,
                        "seed": effective_seed,
                        "quality": mapped.quality.to_dict(),
                        "failures": list(mapped.quality_failures),
                        "advisories": list(
                            mapped.quality_policy.advisories(mapped.quality)
                        ),
                    }
                )
                if scene_or_prompt_changed and self._previous_image is not None:
                    final_difference = image_mean_absolute_difference(
                        self._previous_image,
                        current_image,
                    )

            stem = generation.image_path.stem
            sdl_preview_path = run_dir / f"{stem}_sdl_smooth.png"
            sdl_control_path = run_dir / f"{stem}_sdl_smooth_control.png"
            mask_path = run_dir / f"{stem}_out_of_gamut_mask.png"
            mapped.image.save(sdl_preview_path, format="PNG")
            mapped.control_image.save(sdl_control_path, format="PNG")
            mapped.out_of_gamut_mask.save(mask_path, format="PNG")

            quality = mapped.quality.to_dict()
            provenance_path = self.lora_path.parent / "weight_provenance.json"
            weight_provenance: dict[str, Any] = {}
            if provenance_path.is_file():
                try:
                    weight_provenance = json.loads(
                        provenance_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    weight_provenance = {
                        "provenance_status": "unreadable",
                        "path": str(provenance_path),
                    }
            report_path = run_dir / "pipeline_report.json"
            report = {
                "traceability": {
                    "pipeline_schema_version": 4,
                    "base_model": generation_details.get(
                        "base_model",
                        self.base_model,
                    ),
                    "base_model_revision": generation_details.get(
                        "base_model_revision",
                        DEFAULT_BASE_MODEL_REVISION,
                    ),
                    "lora_path": str(self.lora_path),
                    "lora_sha256": generation_details.get(
                        "lora_sha256",
                        _sha256_file(self.lora_path),
                    ),
                    "weight_provenance_path": str(provenance_path),
                    "weight_provenance": weight_provenance,
                    "sdl_table_path": str(self.sdl_path),
                    "sdl_table_sha256": _sha256_file(self.sdl_path),
                },
                "scene": scene,
                "fixture": {
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "space_size_m2": space_size_m2,
                },
                "image_dimensions": {"width": width, "height": height},
                "palette_notice": palette_notice,
                "module_01": attributes_dict,
                "module_01_prompt_source": prompt_source,
                "module_03": {
                    "generation_mode": generation_details.get(
                        "generation_mode",
                        "unknown",
                    ),
                    "raw_image_path": str(generation.image_path),
                    "manifest_path": str(generation.manifest_path),
                    "diffusion_raw_path": (
                        str(generation.diffusion_raw_path)
                        if generation.diffusion_raw_path is not None
                        else None
                    ),
                    "guided_image_path": (
                        str(generation.guided_image_path)
                        if generation.guided_image_path is not None
                        else None
                    ),
                    "raw_image_sha256": _sha256_file(generation.image_path),
                    "seed": effective_seed,
                    "requested_seed": requested_seed,
                    "seed_mode": seed_mode,
                    "steps": steps,
                    "lora_scale": self.lora_scale,
                    "quality": raw_quality,
                    "artifact_cleanup": artifact_cleanup,
                    "prompt_color_guidance": color_guidance,
                    "quality_retry_count": generation.quality_retry_count,
                    "similarity_guard": {
                        "threshold": SIMILARITY_DIFFERENCE_THRESHOLD,
                        "compared_with_previous": scene_or_prompt_changed,
                        "initial_mean_absolute_difference": initial_difference,
                        "retry_count": retry_count,
                        "final_mean_absolute_difference": final_difference,
                    },
                },
                "module_05": {
                    "enabled": bool(pattern_enabled),
                    "input_raw_path": str(generation.image_path),
                    "input_raw_sha256": _sha256_file(generation.image_path),
                    "themed_image_path": str(pattern.image_path),
                    "themed_image_sha256": _sha256_file(pattern.image_path),
                    "manifest_path": str(pattern.manifest_path),
                    "attributes": pattern.attributes.to_dict(),
                    "render": pattern.render_report,
                },
                "module_04": {
                    "method": "smooth",
                    "preview_algorithm": (
                        "continuous_xy_projection_out_of_gamut_only"
                    ),
                    "control_algorithm": "strict_sdl_ordered_dither",
                    "sdl_preview_path": str(sdl_preview_path),
                    "sdl_control_path": str(sdl_control_path),
                    "out_of_gamut_mask_path": str(mask_path),
                    "sdl_preview_sha256": _sha256_file(sdl_preview_path),
                    "sdl_control_sha256": _sha256_file(sdl_control_path),
                    "input_raw_sha256": _sha256_file(pattern.image_path),
                    "input_source": "module_05_themed_image",
                    "quality": quality,
                    "quality_policy": asdict(mapped.quality_policy),
                    "quality_status": (
                        "accepted" if mapped.accepted else "rejected"
                    ),
                    "quality_failures": list(mapped.quality_failures),
                    "quality_advisories": list(
                        mapped.quality_policy.advisories(mapped.quality)
                    ),
                    "retry_count": sdl_retry_count,
                    "attempts": sdl_attempts,
                },
            }
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if mapped.quality_failures:
                raise RuntimeError(
                    "SDL mapped result failed quality policy after "
                    f"{sdl_retry_count + 1} attempts: "
                    + "; ".join(mapped.quality_failures)
                )

            self._previous_image = current_image.copy()
            self._previous_scene = scene
            self._previous_prompt = attributes.effect

            self._notify(progress, 0.95, "正在整理下载文件")
            archive_path = run_dir / "lighting_demo_result.zip"
            with zipfile.ZipFile(
                archive_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive_paths = [
                    prompt_json_path,
                    generation.image_path,
                    generation.manifest_path,
                    pattern.image_path,
                    pattern.manifest_path,
                    sdl_preview_path,
                    sdl_control_path,
                    mask_path,
                    report_path,
                ]
                for optional_path in (
                    generation.diffusion_raw_path,
                    generation.guided_image_path,
                ):
                    if optional_path is not None:
                        archive_paths.append(optional_path)
                for path in archive_paths:
                    archive.write(path, arcname=path.name)

            self._notify(progress, 1.0, "完成")
            return DemoPipelineResult(
                run_dir=run_dir,
                prompt=attributes.effect,
                attributes=attributes_dict,
                raw_image_path=generation.image_path,
                themed_image_path=pattern.image_path,
                sdl_preview_path=sdl_preview_path,
                sdl_control_path=sdl_control_path,
                out_of_gamut_mask_path=mask_path,
                prompt_json_path=prompt_json_path,
                generation_manifest_path=generation.manifest_path,
                pattern_manifest_path=pattern.manifest_path,
                report_path=report_path,
                archive_path=archive_path,
                quality=quality,
                raw_quality=raw_quality,
                artifact_cleanup=artifact_cleanup,
                color_guidance=color_guidance,
                pattern_report=pattern.render_report,
                palette_notice=palette_notice,
                width=width,
                height=height,
                requested_seed=requested_seed,
                effective_seed=effective_seed,
                seed_mode=seed_mode,
                quality_retry_count=generation.quality_retry_count,
                similarity_retry_count=retry_count,
                similarity_difference=final_difference,
                sdl_retry_count=sdl_retry_count,
            )
