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
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Sequence
from uuid import uuid4
import zipfile

import numpy as np
from PIL import Image
from PIL import ImageDraw

from modules.module_01_prompt_agent.src.agent import LightingPromptAgent
from modules.module_01_prompt_agent.src.fast_compiler import FastPromptCompiler
from modules.module_01_prompt_agent.src.schemas import LightingEffectAttributes
from modules.module_03_image_generation.src.config import (
    DEFAULT_NEGATIVE_PROMPT,
    GenerationConfig,
)
from modules.module_03_image_generation.src.concept_palette import (
    build_concept_palette_plan,
    harmonize_concept_image,
    render_shared_palette_gradient,
)
from modules.module_03_image_generation.src.generator import (
    GenerationResult,
    LightEffectGenerator,
)
from modules.module_03_image_generation.src.image_geometry import dimensions_from_fixture
from modules.module_03_image_generation.src.model_loader import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_LORA_PATH,
)
from modules.module_03_image_generation.src.quality import analyze_image_quality
from modules.module_03_image_generation.src.structured_gradient import (
    StructuredGradientPlan,
)
from modules.module_04_gamut_mapping.src.mapper import GamutMapper
from modules.module_04_gamut_mapping.src.sdl_palette import (
    DEFAULT_SDL_PATH,
    SDLPalette,
)
from modules.module_05_pattern_generation.src.generator import PatternGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "demo"
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
    """All requested hues are accepted; SDL mapping handles hardware limits later."""

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
    concept_image_path: Path
    concept_manifest_path: Path
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
    quality: dict[str, Any]
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
    sdl_available: bool
    sdl_notice: str
    timings: dict[str, float]
    deadline_met: bool


@dataclass(frozen=True, slots=True)
class UnavailableSDLQuality:
    """Explicit non-metrics used when the private SDL table is unavailable."""

    pixel_count: int
    status: str = "skipped_missing_sdl_table"
    mapping_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UnavailableSDLPolicy:
    """Policy placeholder that keeps reporting code uniform in preview mode."""

    reason: str

    def advisories(self, _quality: Any) -> tuple[str, ...]:
        return (self.reason,)


@dataclass(frozen=True, slots=True)
class UnavailableSDLResult:
    """Pass-through preview and unmistakable placeholders for missing SDL data."""

    image: Image.Image
    control_image: Image.Image
    out_of_gamut_mask: Image.Image
    quality: UnavailableSDLQuality
    quality_policy: UnavailableSDLPolicy
    quality_failures: tuple[str, ...] = ()
    method: str = "unavailable"
    available: bool = False

    @property
    def accepted(self) -> bool:
        return False


def unavailable_sdl_result(image: Image.Image, sdl_path: Path) -> UnavailableSDLResult:
    """Return module-three/five preview output without claiming SDL compliance."""

    preview = image.convert("RGB").copy()
    width, height = preview.size
    control = Image.new("RGB", preview.size, (42, 42, 46))
    draw = ImageDraw.Draw(control)
    line_width = max(2, min(width, height) // 80)
    draw.line((0, 0, width, height), fill=(180, 84, 84), width=line_width)
    draw.line((0, height, width, 0), fill=(180, 84, 84), width=line_width)
    message = "SDL TABLE REQUIRED"
    text_box = draw.textbbox((0, 0), message)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    draw.rectangle(
        (
            max(0, (width - text_width) // 2 - 10),
            max(0, (height - text_height) // 2 - 8),
            min(width, (width + text_width) // 2 + 10),
            min(height, (height + text_height) // 2 + 8),
        ),
        fill=(42, 42, 46),
    )
    draw.text(
        ((width - text_width) // 2, (height - text_height) // 2),
        message,
        fill=(245, 230, 230),
    )
    mask = Image.new("L", preview.size, 0)
    reason = (
        "SDL mapping was skipped because the private color table is missing: "
        f"{sdl_path}"
    )
    return UnavailableSDLResult(
        image=preview,
        control_image=control,
        out_of_gamut_mask=mask,
        quality=UnavailableSDLQuality(pixel_count=width * height),
        quality_policy=UnavailableSDLPolicy(reason=reason),
    )


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
        fast_prompt_compiler: Any | None = None,
        lora_scale: float = 0.8,
        allow_missing_sdl: bool = True,
        fast_mode: bool = False,
        time_budget_seconds: float = 6.0,
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
        self.fast_prompt_compiler = fast_prompt_compiler or FastPromptCompiler()
        self.lora_scale = lora_scale
        self.allow_missing_sdl = bool(allow_missing_sdl)
        self.fast_mode = bool(fast_mode)
        self.time_budget_seconds = float(time_budget_seconds)
        self._generator: Any | None = None
        self._mapper: Any | None = None
        self._run_lock = threading.Lock()
        self._previous_image: Image.Image | None = None
        self._previous_scene: str | None = None
        self._previous_prompt: str | None = None

    def warmup(self) -> None:
        """Load models, the fast concept adapter, and SDL data before the first click."""

        generator = self._get_generator()
        self._get_mapper()
        if self.fast_mode and hasattr(generator, "generate_concept"):
            with tempfile.TemporaryDirectory(prefix="lighting-concept-warmup-") as directory:
                generator.generate_concept(
                    "A bright modern interior with plants, furniture, natural depth, "
                    "and balanced cinematic ambient light, no text or logos.",
                    output_dir=Path(directory),
                    seed=1,
                    steps=2,
                    width=256,
                    height=144,
                )

    def _generate_fast_light(
        self,
        *,
        shared_plan: StructuredGradientPlan,
        palette_report: dict[str, object],
        output_dir: Path,
        config: GenerationConfig,
        source_attributes: dict[str, Any],
    ) -> GenerationResult:
        image, color_guidance = render_shared_palette_gradient(
            shared_plan,
            palette_report,
            width=config.width,
            height=config.height,
        )
        quality = analyze_image_quality(image)
        image_path = output_dir / f"raw_light_effect_seed_{config.seed}.png"
        manifest_path = image_path.with_suffix(".json")
        image.save(image_path, format="PNG", optimize=False)
        manifest_path.write_text(
            json.dumps(
                {
                    "prompt": source_attributes.get("effect", ""),
                    "effective_prompt": source_attributes.get("effect", ""),
                    "base_model": self.base_model,
                    "base_model_revision": DEFAULT_BASE_MODEL_REVISION,
                    "lora_path": str(self.lora_path),
                    "lora_sha256": _sha256_file(self.lora_path),
                    "device": self.device,
                    "module_01_attributes": source_attributes,
                    "generation_mode": "concept_palette_fast",
                    "prompt_color_guidance": color_guidance,
                    "artifact_cleanup": {"applied": False},
                    "quality_status": "accepted",
                    "quality_retry_count": 0,
                    "quality": quality.to_dict(),
                    "seed": config.seed,
                    "width": config.width,
                    "height": config.height,
                    "num_inference_steps": 0,
                    "image_path": str(image_path),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return GenerationResult(
            image_path=image_path,
            manifest_path=manifest_path,
            seed=config.seed,
            width=config.width,
            height=config.height,
            quality_retry_count=0,
        )

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

    def _get_mapper(self) -> Any | None:
        if not self.sdl_path.is_file():
            if self.allow_missing_sdl:
                return None
            raise FileNotFoundError(
                "SDL color table does not exist: "
                f"{self.sdl_path}. Copy the authorized table to this path or "
                "start the demo without --require-sdl to use preview mode."
            )
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
        forbidden_prompts: Sequence[str] = (),
        forbidden_prompt_designs: Sequence[str] = (),
        pattern_enabled: bool = True,
        pattern_strength: float | None = None,
        progress: Callable[[float, str], Any] | None = None,
    ) -> DemoPipelineResult:
        """Run modules 1, 3, and 4 and persist every intermediate artifact."""

        pipeline_started = time.perf_counter()
        timings: dict[str, float] = {}
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
            stage_started = time.perf_counter()
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
                prompt_compiler = (
                    self.fast_prompt_compiler
                    if self.fast_mode
                    else self._get_prompt_agent()
                )
                attributes = prompt_compiler.generate(
                    scene,
                    hardware_width_mm=width_mm,
                    hardware_height_mm=height_mm,
                    space_size_m2=space_size_m2,
                    forbidden_effects=forbidden_prompts,
                    forbidden_design_effects=forbidden_prompt_designs,
                )
                prompt_source = "local_fast_compiler" if self.fast_mode else "deepseek"
            timings["prompt_seconds"] = time.perf_counter() - stage_started
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

            generator = self._get_generator()
            if self.fast_mode:
                self._notify(progress, 0.20, "正在快速生成实体场景概念图")
                stage_started = time.perf_counter()
                concept_prompt = attributes.concept_prompt or (
                    "Cinematic real-world interior scene with recognizable objects, "
                    "people, realistic materials, natural depth, and atmospheric "
                    f"illumination inspired by {attributes.effect}"
                )
                concept = generator.generate_concept(
                    concept_prompt,
                    output_dir=run_dir,
                    seed=effective_seed,
                    steps=steps,
                )
                timings["concept_seconds"] = time.perf_counter() - stage_started
                with Image.open(concept.image_path) as opened:
                    source_concept_image = opened.convert("RGB").copy()
                stage_started = time.perf_counter()
                shared_plan, palette_report = build_concept_palette_plan(
                    source_concept_image,
                    attributes.effect,
                )
                concept_image, harmonization_report = harmonize_concept_image(
                    source_concept_image,
                    shared_plan,
                )
                concept_source_image_path = concept.image_path
                concept_image_path = run_dir / (
                    f"concept_image_harmonized_seed_{effective_seed}.png"
                )
                concept_image.save(concept_image_path, format="PNG", optimize=False)
                concept_manifest_path = concept.manifest_path
                concept_details = json.loads(
                    concept_manifest_path.read_text(encoding="utf-8")
                )
                concept_details.update(
                    {
                        "source_image_path": str(concept_source_image_path),
                        "display_image_path": str(concept_image_path),
                        "shared_palette": palette_report,
                        "harmonization": harmonization_report,
                    }
                )
                concept_manifest_path.write_text(
                    json.dumps(concept_details, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                timings["concept_harmonization_seconds"] = (
                    time.perf_counter() - stage_started
                )
                self._notify(progress, 0.56, "正在复用共享色彩蓝图生成光色图")
                stage_started = time.perf_counter()
                generation = self._generate_fast_light(
                    shared_plan=shared_plan,
                    palette_report=palette_report,
                    output_dir=run_dir,
                    config=config,
                    source_attributes=attributes_dict,
                )
                timings["light_field_seconds"] = time.perf_counter() - stage_started
            else:
                self._notify(
                    progress,
                    0.30,
                    "模块三：正在生成主渐变并提取弱 LoRA 亮度纹理",
                )
                stage_started = time.perf_counter()
                generation = generator.generate(
                    attributes.effect,
                    output_dir=run_dir,
                    config=config,
                    source_attributes=attributes_dict,
                )
                timings["light_field_seconds"] = time.perf_counter() - stage_started
                concept_image_path = generation.image_path
                concept_source_image_path = generation.image_path
                concept_manifest_path = generation.manifest_path
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
                    and not self.fast_mode
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
            if not self.fast_mode:
                concept_image_path = generation.image_path
                concept_source_image_path = generation.image_path
                concept_manifest_path = generation.manifest_path
            raw_quality = dict(generation_details.get("quality", {}))
            artifact_cleanup = dict(
                generation_details.get("artifact_cleanup", {})
            )
            color_guidance = dict(
                generation_details.get("prompt_color_guidance", {})
            )

            self._notify(progress, 0.73, "模块五：正在进行低频主题光场增强")
            stage_started = time.perf_counter()
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
            timings["pattern_seconds"] = time.perf_counter() - stage_started

            mapper = self._get_mapper()
            stage_started = time.perf_counter()
            if mapper is None:
                self._notify(
                    progress,
                    0.78,
                    "模块四：未配置 SDL 色表，正在输出模块三/五预览",
                )
                mapped = unavailable_sdl_result(mapping_image, self.sdl_path)
            else:
                self._notify(progress, 0.78, "模块四：正在进行 SDL 色域映射")
                mapped = mapper.map_image(mapping_image, method="smooth")
            timings["sdl_seconds"] = time.perf_counter() - stage_started
            sdl_available = bool(getattr(mapped, "available", True))
            sdl_notice = (
                ""
                if sdl_available
                else (
                    "未找到本地 SDL 色表，模块四已跳过；"
                    "当前仅输出模块三 Raw 和模块五主题预览，"
                    "不代表硬件可实现色域。"
                )
            )
            if sdl_available and self.fast_mode and mapped.quality_failures:
                sdl_notice = (
                    "快速模式未执行耗时重试；视觉预览已返回，严格控制图仍保持 "
                    "SDL 表成员合规。建议在质量模式复核："
                    + ", ".join(mapped.quality_failures)
                )
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
            if (
                sdl_available
                and mapped.quality_failures
                and MAX_SDL_RETRIES
                and not fixed_seed
                and not self.fast_mode
            ):
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

            if not self.fast_mode:
                concept_image_path = generation.image_path
                concept_source_image_path = generation.image_path
                concept_manifest_path = generation.manifest_path

            stem = generation.image_path.stem
            suffix = "sdl_smooth" if sdl_available else "sdl_unavailable_preview"
            sdl_preview_path = run_dir / f"{stem}_{suffix}.png"
            sdl_control_path = run_dir / f"{stem}_{suffix}_control.png"
            mask_path = run_dir / f"{stem}_{suffix}_mask.png"
            mapped.image.save(sdl_preview_path, format="PNG")
            mapped.control_image.save(sdl_control_path, format="PNG")
            mapped.out_of_gamut_mask.save(mask_path, format="PNG")

            quality = mapped.quality.to_dict()
            timings["pre_export_seconds"] = time.perf_counter() - pipeline_started
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
                    "pipeline_schema_version": 6,
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
                    "sdl_table_sha256": (
                        _sha256_file(self.sdl_path)
                        if self.sdl_path.is_file()
                        else None
                    ),
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
                "concept_image": {
                    "path": str(concept_image_path),
                    "source_path": str(concept_source_image_path),
                    "manifest_path": str(concept_manifest_path),
                    "sha256": _sha256_file(concept_image_path),
                    "prompt": attributes.concept_prompt,
                },
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
                    "available": sdl_available,
                    "notice": sdl_notice,
                    "method": "smooth" if sdl_available else None,
                    "preview_algorithm": (
                        "continuous_xy_projection_out_of_gamut_only"
                        if sdl_available
                        else "module_05_passthrough_preview"
                    ),
                    "control_algorithm": (
                        "strict_sdl_ordered_dither"
                        if sdl_available
                        else None
                    ),
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
                        "accepted"
                        if sdl_available and mapped.accepted
                        else (
                            "rejected"
                            if sdl_available
                            else "skipped_missing_sdl_table"
                        )
                    ),
                    "quality_failures": list(mapped.quality_failures),
                    "quality_advisories": list(
                        mapped.quality_policy.advisories(mapped.quality)
                    ),
                    "retry_count": sdl_retry_count,
                    "attempts": sdl_attempts,
                },
                "performance": {
                    "mode": "fast_dual_image" if self.fast_mode else "quality",
                    "budget_seconds": self.time_budget_seconds,
                    "timings": dict(timings),
                },
            }
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if mapped.quality_failures and not self.fast_mode:
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
                    concept_image_path,
                    concept_source_image_path,
                    concept_manifest_path,
                    generation.image_path,
                    generation.manifest_path,
                    pattern.image_path,
                    pattern.manifest_path,
                    sdl_preview_path,
                    sdl_control_path,
                    mask_path,
                ]
                for optional_path in (
                    generation.diffusion_raw_path,
                    generation.guided_image_path,
                ):
                    if optional_path is not None:
                        archive_paths.append(optional_path)
                unique_archive_paths = list(dict.fromkeys(archive_paths))
                for path in unique_archive_paths:
                    archive.write(path, arcname=path.name)

                timings["total_seconds"] = time.perf_counter() - pipeline_started
                deadline_met = timings["total_seconds"] <= self.time_budget_seconds
                report["performance"] = {
                    "mode": "fast_dual_image" if self.fast_mode else "quality",
                    "budget_seconds": self.time_budget_seconds,
                    "deadline_met": deadline_met,
                    "timings": dict(timings),
                }
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                archive.write(report_path, arcname=report_path.name)

            self._notify(progress, 1.0, "完成")
            return DemoPipelineResult(
                run_dir=run_dir,
                prompt=attributes.effect,
                attributes=attributes_dict,
                concept_image_path=concept_image_path,
                concept_manifest_path=concept_manifest_path,
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
                sdl_available=sdl_available,
                sdl_notice=sdl_notice,
                timings=timings,
                deadline_met=deadline_met,
            )
