"""Validated inference configuration for raw light-effect generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_NEGATIVE_PROMPT = (
    "text, letters, logo, watermark, people, person, face, body, furniture, "
    "lamp, fixture, architecture, objects, hard edges, sharp shapes, noise, "
    "grain, isolated spots, speckles, colored dots, small blobs, clusters, "
    "streaks, narrow beams, hard lines, low resolution, black background"
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Parameters needed to reproduce one generated image."""

    width: int = 1024
    height: int = 320
    seed: int = 20260719
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    lora_scale: float = 1.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

    def __post_init__(self) -> None:
        for name, value in (("width", self.width), ("height", self.height)):
            if value < 64 or value % 8:
                raise ValueError(f"{name} must be at least 64 and divisible by 8")
        if not 1 <= self.num_inference_steps <= 150:
            raise ValueError("num_inference_steps must be from 1 to 150")
        if not 0 <= self.guidance_scale <= 30:
            raise ValueError("guidance_scale must be from 0 to 30")
        if not 0 <= self.lora_scale <= 2:
            raise ValueError("lora_scale must be from 0 to 2")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be from 0 to 2**63 - 1")
        if not self.negative_prompt.strip():
            raise ValueError("negative_prompt cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
