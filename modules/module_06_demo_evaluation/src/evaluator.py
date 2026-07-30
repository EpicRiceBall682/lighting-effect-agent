"""Resumable batch evaluation over the organizer's test scenes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from .pipeline import LightingDemoPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEST_SET = PROJECT_ROOT / "reference_data" / "测试集.txt"
DEFAULT_BATCH_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation"
SIGNATURE_SOURCE_FILES = (
    PROJECT_ROOT / "modules" / "module_01_prompt_agent" / "src" / "agent.py",
    PROJECT_ROOT / "modules" / "module_03_image_generation" / "src" / "generator.py",
    PROJECT_ROOT
    / "modules"
    / "module_03_image_generation"
    / "src"
    / "structured_gradient.py",
    PROJECT_ROOT / "modules" / "module_04_gamut_mapping" / "src" / "mapper.py",
    PROJECT_ROOT
    / "modules"
    / "module_05_pattern_generation"
    / "src"
    / "generator.py",
    PROJECT_ROOT / "modules" / "module_06_demo_evaluation" / "src" / "pipeline.py",
)


@dataclass(frozen=True, slots=True)
class BatchEvaluationConfig:
    """Shared fixture and inference settings for a batch."""

    width_mm: float = 1220
    height_mm: float = 370
    space_size_m2: float | None = None
    seed: int = 20260724
    steps: int = 30
    fixed_seed: bool = False


def load_test_scenes(path: Path = DEFAULT_TEST_SET) -> list[str]:
    """Load non-empty, trimmed scenes while preserving organizer order."""

    scenes = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not scenes:
        raise ValueError(f"test set contains no scenes: {path}")
    return scenes


def _case_id(index: int, scene: str) -> str:
    normalized = " ".join(scene.casefold().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{index:04d}-{digest}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_signature(
    pipeline: Any,
    config: BatchEvaluationConfig,
) -> str:
    """Fingerprint settings, runtime assets, and core source code for safe resume."""

    pipeline_settings: dict[str, Any] = {
        "pipeline_type": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}",
    }
    for name in ("device", "base_model", "lora_scale"):
        if hasattr(pipeline, name):
            pipeline_settings[name] = getattr(pipeline, name)
    for name in ("lora_path", "sdl_path"):
        value = getattr(pipeline, name, None)
        if value is None:
            continue
        path = Path(value).expanduser().resolve()
        pipeline_settings[name] = str(path)
        pipeline_settings[f"{name}_sha256"] = (
            _sha256_file(path) if path.is_file() else None
        )

    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in SIGNATURE_SOURCE_FILES
        if path.is_file()
    }
    payload = {
        "schema_version": 1,
        "config": asdict(config),
        "pipeline": pipeline_settings,
        "source_sha256": source_hashes,
    }
    normalized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _result_artifacts_exist(record: dict[str, Any]) -> bool:
    required = (
        "raw_image_path",
        "sdl_preview_path",
        "sdl_control_path",
        "report_path",
        "archive_path",
    )
    return all(
        record.get(name) and Path(str(record[name])).expanduser().is_file()
        for name in required
    )


def _read_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise ValueError(f"{path}:{line_number} has no case_id")
        records[case_id] = record
    return records


def _append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def run_batch_evaluation(
    pipeline: Any,
    scenes: Iterable[str],
    output_dir: Path,
    *,
    config: BatchEvaluationConfig = BatchEvaluationConfig(),
    resume: bool = True,
) -> dict[str, Any]:
    """Run all scenes, persist each result immediately, and continue on errors."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "batch_results.jsonl"
    existing = _read_results(results_path) if resume else {}
    if not resume:
        results_path.write_text("", encoding="utf-8")

    normalized_scenes = [str(scene).strip() for scene in scenes if str(scene).strip()]
    if not normalized_scenes:
        raise ValueError("at least one non-empty scene is required")
    evaluation_signature = _evaluation_signature(pipeline, config)

    completed: list[dict[str, Any]] = []
    newly_processed = 0
    reused = 0
    for index, scene in enumerate(normalized_scenes, 1):
        case_id = _case_id(index, scene)
        previous = existing.get(case_id)
        if (
            previous
            and previous.get("status") == "success"
            and previous.get("evaluation_signature") == evaluation_signature
            and _result_artifacts_exist(previous)
        ):
            completed.append(previous)
            reused += 1
            continue

        base_record: dict[str, Any] = {
            "case_id": case_id,
            "case_index": index,
            "scene": scene,
            "fixture": {
                "width_mm": config.width_mm,
                "height_mm": config.height_mm,
                "space_size_m2": config.space_size_m2,
            },
            "requested_seed": config.seed,
            "steps": config.steps,
            "fixed_seed": config.fixed_seed,
            "evaluation_signature": evaluation_signature,
        }
        try:
            result = pipeline.run(
                scene,
                config.width_mm,
                config.height_mm,
                config.space_size_m2,
                seed=config.seed,
                steps=config.steps,
                fixed_seed=config.fixed_seed,
            )
            record = {
                **base_record,
                "status": "success",
                "english_prompt": result.prompt,
                # The current prompt agent emits English only. Preserve the
                # organizer's Chinese scene verbatim instead of inventing a
                # machine translation.
                "chinese_prompt": scene,
                "attributes": result.attributes,
                "raw_image_path": str(result.raw_image_path),
                "themed_image_path": (
                    str(result.themed_image_path)
                    if getattr(result, "themed_image_path", None) is not None
                    else None
                ),
                "sdl_preview_path": str(result.sdl_preview_path),
                "sdl_control_path": str(result.sdl_control_path),
                "report_path": str(result.report_path),
                "archive_path": str(result.archive_path),
                "effective_seed": result.effective_seed,
                "seed_mode": result.seed_mode,
                "raw_quality": result.raw_quality,
                "color_guidance": getattr(result, "color_guidance", {}),
                "pattern_report": getattr(result, "pattern_report", {}),
                "sdl_quality": result.quality,
                "quality_retry_count": int(
                    getattr(result, "quality_retry_count", 0)
                ),
                "similarity_retry_count": result.similarity_retry_count,
                "similarity_difference": result.similarity_difference,
                "sdl_retry_count": int(
                    getattr(result, "sdl_retry_count", 0)
                ),
            }
        except Exception as exc:  # keep the batch running and make failure auditable
            record = {
                **base_record,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        _append_record(results_path, record)
        existing[case_id] = record
        completed.append(record)
        newly_processed += 1

    completed.sort(key=lambda item: int(item["case_index"]))
    succeeded = sum(item["status"] == "success" for item in completed)
    failed = len(completed) - succeeded
    summary = {
        "total": len(completed),
        "succeeded": succeeded,
        "failed": failed,
        "newly_processed": newly_processed,
        "reused_successes": reused,
        "complete": failed == 0 and len(completed) == len(normalized_scenes),
        "results_path": str(results_path),
        "failed_cases": [
            {
                "case_id": item["case_id"],
                "scene": item["scene"],
                "error": item.get("error", ""),
            }
            for item in completed
            if item["status"] != "success"
        ],
    }
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run resumable batch lighting evaluation.")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_OUTPUT)
    parser.add_argument("--width-mm", type=float, default=1220)
    parser.add_argument("--height-mm", type=float, default=370)
    parser.add_argument("--space-size-m2", type=float)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fixed-seed", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_batch_evaluation(
            LightingDemoPipeline(output_root=args.output_dir / "runs"),
            load_test_scenes(args.test_set),
            args.output_dir,
            config=BatchEvaluationConfig(
                width_mm=args.width_mm,
                height_mm=args.height_mm,
                space_size_m2=args.space_size_m2,
                seed=args.seed,
                steps=args.steps,
                fixed_seed=args.fixed_seed,
            ),
            resume=not args.no_resume,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
