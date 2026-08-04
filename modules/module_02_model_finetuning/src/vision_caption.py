"""Create image-grounded synthetic captions with local or remote vision models."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


DEFAULT_ENDPOINT = "https://models.github.ai/inference/chat/completions"
DEFAULT_GITHUB_MODEL = "openai/gpt-4o"
DEFAULT_LOCAL_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
FORBIDDEN_CAPTION_TERMS = (
    "black",
    "shadow",
)


class VisionCaptionError(RuntimeError):
    """Raised when a vision caption request or response is invalid."""


class CaptionClient(Protocol):
    model: str
    provider: str

    def caption(self, image_path: Path) -> str: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_caption(text: str) -> str:
    """Keep captions concise, English-only and inside the allowed color vocabulary."""

    text = " ".join(text.strip().split())
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)
    if not 12 <= len(words) <= 55 or re.search(r"[\u3400-\u9fff]", text):
        raise VisionCaptionError("caption must be one English sentence of 12 to 55 words")
    lowered = text.casefold()
    forbidden = [
        term for term in FORBIDDEN_CAPTION_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)
    ]
    if forbidden:
        raise VisionCaptionError(f"caption contains forbidden colors: {', '.join(forbidden)}")
    return text


def _image_data_url(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 512), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class GitHubVisionCaptionClient:
    """Minimal resumable-friendly client for GPT-4o image captions."""

    def __init__(
        self,
        *,
        token: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        self.model = model or os.getenv("GITHUB_VISION_MODEL", DEFAULT_GITHUB_MODEL)
        self.provider = "github_models"
        self.endpoint = endpoint or os.getenv("GITHUB_MODELS_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        if not self.token:
            raise VisionCaptionError("missing token; set GITHUB_TOKEN before captioning images")
        if not self.endpoint.startswith("https://"):
            raise VisionCaptionError("GitHub Models endpoint must use HTTPS")

    def caption(self, image_path: Path) -> str:
        prompt = (
            "Inspect this synthetic panoramic light-effect texture. Return one JSON object with "
            "exactly one field named text. The text must be a single English sentence of 20 to 45 "
            "words describing only what is visible: color order and direction, gradient structure, "
            "mist, clouds or glow, brightness and smoothness. Do not invent a room, people or objects. "
            "All visible hue families are allowed, including green, cyan, teal, navy and indigo. "
            "Use precise color names and preserve their actual left-to-right order."
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image_path), "detail": "low"},
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 180,
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "lighting-effect-vision-captioner",
            },
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                content = response_data["choices"][0]["message"]["content"]
                decoded = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
                if set(decoded) != {"text"}:
                    raise VisionCaptionError("vision model JSON must contain exactly the text field")
                return validate_caption(str(decoded["text"]))
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")[:800]
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise VisionCaptionError(
                        f"GitHub vision request failed with HTTP {exc.code}: {error_body}"
                    ) from exc
            except (URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise VisionCaptionError(f"GitHub vision request failed: {exc}") from exc
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise VisionCaptionError("vision model returned malformed JSON") from exc
            time.sleep(2**attempt)
        raise VisionCaptionError("vision model returned no caption")


class LocalSmolVLMCaptionClient:
    """Run a small open vision-language model locally on Apple MPS, CUDA or CPU."""

    def __init__(self, *, model: str = DEFAULT_LOCAL_MODEL, device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if device not in {"auto", "mps", "cuda", "cpu"}:
            raise VisionCaptionError("local device must be one of: auto, mps, cuda, cpu")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise VisionCaptionError("CUDA was requested but is not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise VisionCaptionError("MPS was requested but is not available")

        self.model = model
        self.provider = "local_smolvlm"
        self.device = device
        self.torch = torch
        self.dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
        try:
            self.processor = AutoProcessor.from_pretrained(model)
            self.vision_model = AutoModelForMultimodalLM.from_pretrained(
                model,
                torch_dtype=self.dtype,
                _attn_implementation="eager",
            ).to(device)
            self.vision_model.eval()
        except Exception as exc:
            raise VisionCaptionError(f"failed to load local vision model {model}: {exc}") from exc

    def caption(self, image_path: Path) -> str:
        prompt = (
            "Describe this abstract panoramic lighting texture in one English sentence of 20 to "
            "45 words. Mention only the visible color order and direction, gradient structure, "
            "mist, clouds or glow, brightness and smoothness. Use only these color terms when "
            "appropriate: white, ivory, pale yellow, amber, orange, coral, pink, red, lavender, "
            "light blue. Do not invent a room, landscape, people or objects."
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            rendered_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(text=rendered_prompt, images=[image], return_tensors="pt")
            for key, value in inputs.items():
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    inputs[key] = value.to(self.device, dtype=self.dtype)
                else:
                    inputs[key] = value.to(self.device)
            with self.torch.inference_mode():
                generated = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            text = self.processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )[0]
            return validate_caption(text)
        except VisionCaptionError:
            raise
        except Exception as exc:
            raise VisionCaptionError(f"local vision caption failed for {image_path.name}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def caption_metadata(
    source_metadata: Path,
    output_metadata: Path,
    client: CaptionClient | None,
    *,
    cache_metadata_paths: tuple[Path, ...] = (),
    limit: int | None = None,
    request_interval_seconds: float = 0.0,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    """Caption records one-by-one and save after every success so runs can resume."""

    if source_metadata.resolve() == output_metadata.resolve():
        raise ValueError("output metadata must be different from source metadata")
    source_records = _read_jsonl(source_metadata)
    cache_records: list[dict[str, Any]] = []
    if output_metadata.exists():
        cache_records.extend(_read_jsonl(output_metadata))
    for cache_path in cache_metadata_paths:
        if cache_path.exists():
            cache_records.extend(_read_jsonl(cache_path))
    existing = {record["file_name"]: record for record in cache_records}
    existing_by_hash = {
        record["caption_image_sha256"]: record
        for record in cache_records
        if record.get("caption_source") == "vision_model"
        and record.get("caption_image_sha256")
    }
    completed: list[dict[str, Any]] = []
    newly_captioned = 0
    reused = 0
    remaining_uncaptioned = 0

    for source_record in source_records:
        image_path = source_metadata.parent / source_record["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"image referenced by metadata does not exist: {image_path}")
        image_sha256 = sha256_file(image_path)
        cached = existing.get(source_record["file_name"])
        if not cached or cached.get("caption_image_sha256") != image_sha256:
            cached = existing_by_hash.get(image_sha256)
        if (
            cached
            and cached.get("caption_source") == "vision_model"
            and cached.get("caption_image_sha256") == image_sha256
        ):
            reused_record = dict(source_record)
            reused_record.update(
                {
                    "template_text": source_record.get(
                        "template_text", source_record.get("text", "")
                    ),
                    "text": cached["text"],
                    "caption_source": "vision_model",
                    "caption_model": cached.get("caption_model", ""),
                    "caption_provider": cached.get("caption_provider", ""),
                    "caption_image_sha256": image_sha256,
                }
            )
            completed.append(reused_record)
            reused += 1
            continue
        if limit is not None and newly_captioned >= limit:
            completed.append(source_record)
            remaining_uncaptioned += 1
            continue
        if client is None:
            completed.append(source_record)
            remaining_uncaptioned += 1
            continue

        text = client.caption(image_path)
        updated = dict(source_record)
        updated.update(
            {
                "template_text": source_record.get("template_text", source_record.get("text", "")),
                "text": text,
                "caption_source": "vision_model",
                "caption_model": client.model,
                "caption_provider": getattr(client, "provider", "unknown"),
                "caption_image_sha256": image_sha256,
            }
        )
        completed.append(updated)
        newly_captioned += 1
        _write_jsonl_atomic(output_metadata, completed + source_records[len(completed) :])
        if progress:
            progress(len(completed), len(source_records), source_record["file_name"])
        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

    _write_jsonl_atomic(output_metadata, completed)
    return {
        "total": len(source_records),
        "newly_captioned": newly_captioned,
        "reused": reused,
        "remaining_uncaptioned": remaining_uncaptioned,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caption synthetic images with a vision model.")
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument(
        "--cache-metadata",
        type=Path,
        action="append",
        default=[],
        help="reuse vision captions for identical image hashes, even if filenames changed",
    )
    parser.add_argument(
        "--provider",
        choices=("local", "github"),
        default="local",
        help="local is recommended because GitHub Models retires on 2026-07-30",
    )
    parser.add_argument("--model", help="override the provider's default model")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="reuse matching captions without loading or calling a vision model",
    )
    parser.add_argument("--limit", type=int, help="caption at most this many new images in this run")
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="seconds between images; normally zero for the local provider",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    try:
        if args.cache_only:
            client = None
        elif args.provider == "local":
            client = LocalSmolVLMCaptionClient(
                model=args.model or DEFAULT_LOCAL_MODEL,
                device=args.device,
            )
        else:
            client = GitHubVisionCaptionClient(
                model=args.model or os.getenv("GITHUB_VISION_MODEL", DEFAULT_GITHUB_MODEL)
            )
        summary = caption_metadata(
            args.source_metadata,
            args.output_metadata,
            client,
            cache_metadata_paths=tuple(args.cache_metadata),
            limit=args.limit,
            request_interval_seconds=args.request_interval,
            progress=lambda current, total, name: print(f"[{current}/{total}] 已保存：{name}"),
        )
    except (OSError, ValueError, json.JSONDecodeError, VisionCaptionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.cache_only and summary["remaining_uncaptioned"]:
        print(
            f"error: {summary['remaining_uncaptioned']} images still lack verified visual captions",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
