"""Load the SD 1.5 base model and the trained light-effect LoRA."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any


DEFAULT_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_BASE_MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DEFAULT_LORA_PATH = Path(__file__).resolve().parents[1] / "weights" / "light_effect_lora.safetensors"
DEFAULT_CONCEPT_LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"


class ModelLoadError(RuntimeError):
    """Raised when the inference model cannot be prepared."""


def detect_device(requested: str = "auto") -> str:
    import torch

    requested = requested.lower().strip()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ModelLoadError("MPS was requested but is not available")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    base_model_revision: str = DEFAULT_BASE_MODEL_REVISION,
    lora_path: Path = DEFAULT_LORA_PATH,
    device: str = "auto",
    lora_scale: float = 1.0,
) -> tuple[Any, str]:
    """Return a ready StableDiffusionPipeline and its selected device."""

    import torch
    from diffusers import StableDiffusionPipeline

    lora_path = Path(lora_path).expanduser().resolve()
    if not lora_path.is_file():
        raise ModelLoadError(f"LoRA weight does not exist: {lora_path}")

    selected_device = detect_device(device)
    dtype = torch.float16 if selected_device in {"cuda", "mps"} else torch.float32
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
            revision=base_model_revision,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.load_lora_weights(
            str(lora_path.parent),
            weight_name=lora_path.name,
            adapter_name="light_effect",
        )
        pipeline.set_adapters("light_effect", adapter_weights=lora_scale)
        concept_lcm = os.getenv(
            "CONCEPT_LCM_LORA",
            DEFAULT_CONCEPT_LCM_LORA,
        ).strip()
        pipeline._concept_lcm_available = False
        pipeline._concept_lcm_error = ""
        if concept_lcm:
            try:
                pipeline.load_lora_weights(
                    concept_lcm,
                    adapter_name="concept_lcm",
                )
                pipeline._concept_lcm_available = True
            except Exception as exc:
                # The full-quality path remains usable offline. The concept generator
                # falls back to a short DPM schedule and reports this condition.
                pipeline._concept_lcm_error = str(exc)
        if selected_device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        else:
            # Slicing lowers memory pressure on CPU/MPS, but adds latency on CUDA.
            pipeline.enable_attention_slicing()
            if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
                pipeline.vae.enable_slicing()
            elif hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
        pipeline = pipeline.to(selected_device)
    except Exception as exc:
        raise ModelLoadError(f"failed to load base model or LoRA: {exc}") from exc
    return pipeline, selected_device
