"""Build deterministic train/validation ImageFolder directories for Colab."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any

from PIL import Image

from .audit_dataset import audit_dataset, load_training_rows, sha256_file


def infer_synthetic_grouping_fields(record: dict[str, Any]) -> dict[str, str]:
    """Return grouping metadata, including deterministic legacy-v3 backfills."""

    existing = {
        field: str(record.get(field, "")).strip()
        for field in ("recipe_id", "split_group", "palette_family", "layout_id")
    }
    if all(existing.values()):
        return existing

    source = str(record.get("source", "synthetic"))
    template = str(record.get("template_text") or record.get("text", "")).casefold()
    recipe_id = palette_family = layout_id = ""
    if source == "synthetic_linear":
        match = re.search(
            r"transitioning from (.+?) to (.+?),\s+a soft central mist",
            template,
        )
        if match:
            left_name, right_name = (part.strip() for part in match.groups())
            palette_family = "|".join(sorted((left_name, right_name)))
            recipe_id = f"linear:{left_name}->{right_name}"
            layout_id = "horizontal"
    elif source == "synthetic_fluid":
        mode = str(record.get("generator_mode", "")).strip()
        match = re.search(r"gradient lighting blending (.+?),\s+with soft", template)
        if mode and match:
            names = [part.strip() for part in match.group(1).split(",") if part.strip()]
            palette_family = "|".join(sorted(dict.fromkeys(names)))
            recipe_id = f"fluid:{mode}:{palette_family}"
            layout_id = mode
    elif source == "synthetic_sky":
        style = str(record.get("generator_style", "")).strip()
        layout = str(record.get("generator_layout", "")).strip()
        if style and layout:
            recipe_id = f"sky:{style}:{layout}"
            palette_family = style
            layout_id = layout

    inferred = {
        "recipe_id": recipe_id,
        "split_group": f"{source}:{recipe_id}" if recipe_id else "",
        "palette_family": palette_family,
        "layout_id": layout_id,
    }
    return {
        field: existing[field] or inferred[field]
        for field in existing
    }


def load_synthetic_records(
    metadata_paths: list[Path], *, require_vision_captions: bool = True
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("file_name") or not record.get("text"):
                raise ValueError(f"{metadata_path}:{line_number} requires file_name and text")
            caption_source = str(record.get("caption_source", "template"))
            if require_vision_captions and caption_source != "vision_model":
                raise ValueError(
                    f"{metadata_path}:{line_number} has no vision-model caption; "
                    "run src.vision_caption before preparing the training package"
                )
            source_path = metadata_path.parent / record["file_name"]
            if not source_path.is_file():
                raise FileNotFoundError(f"synthetic image does not exist: {source_path}")
            grouping = infer_synthetic_grouping_fields(record)
            records.append(
                {
                    "source_path": source_path,
                    "text": str(record["text"]).strip(),
                    "scene_prompt": "",
                    "source": str(record.get("source", "synthetic")),
                    "caption_source": caption_source,
                    "caption_model": str(record.get("caption_model", "")),
                    "caption_provider": str(record.get("caption_provider", "")),
                    "original_file_name": source_path.name,
                    **grouping,
                    "extra": {key: value for key, value in record.items() if key not in {"file_name", "text"}},
                }
            )
    return records


def _deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hash[sha256_file(record["source_path"])].append(record)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for digest, group in by_hash.items():
        canonical_source = max(group, key=lambda item: len(item["text"]))
        canonical = dict(canonical_source, content_sha256=digest)
        kept.append(canonical)
        for duplicate in group:
            if duplicate is canonical_source:
                continue
            removed.append(
                {
                    "removed": duplicate["original_file_name"],
                    "kept": canonical["original_file_name"],
                    "content_sha256": digest,
                }
            )
    return kept, removed


def split_records(
    records: list[dict[str, Any]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be greater than 0 and less than 0.5")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for source in sorted(by_source):
        source_records = by_source[source]
        split_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in source_records:
            # Exact-image hashes are the safe fallback for legacy metadata.
            group_key = str(record.get("split_group") or record["content_sha256"])
            split_groups[group_key].append(record)

        groups = sorted(
            split_groups.values(),
            key=lambda items: (
                str(items[0].get("split_group", "")),
                min(str(item["content_sha256"]) for item in items),
            ),
        )
        random.Random(f"{seed}:{source}").shuffle(groups)
        validation_target = (
            max(1, round(len(source_records) * validation_ratio)) if len(groups) > 1 else 0
        )
        validation_groups: list[list[dict[str, Any]]] = []
        validation_size = 0
        # Keep complete recipe/scene groups together and always leave at least
        # one group for training.
        for group_index, group in enumerate(groups[:-1]):
            if validation_size >= validation_target:
                break
            validation_groups.append(group)
            validation_size += len(group)

        selected_ids = {id(group) for group in validation_groups}
        for group in groups:
            destination = validation if id(group) in selected_ids else train
            destination.extend(group)
    return train, validation


def _safe_source_name(source: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in source).strip("_")


def _write_split(split_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_dir.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = defaultdict(int)
    metadata: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (item["source"], item["content_sha256"])):
        source_name = _safe_source_name(record["source"])
        counters[source_name] += 1
        filename = f"{source_name}_{counters[source_name]:04d}.png"
        target = split_dir / filename
        with Image.open(record["source_path"]) as image:
            image.convert("RGB").save(target, format="PNG", optimize=True)
        metadata.append(
            {
                "file_name": filename,
                "text": record["text"],
                "source": record["source"],
                "scene_prompt": record["scene_prompt"],
                # ImageFolder treats every field named *_file_name as another
                # image column. Keep the audit value under a neutral key.
                "original_name": record["original_file_name"],
                "content_sha256": record["content_sha256"],
                "caption_source": record["caption_source"],
                "caption_model": record.get("caption_model", ""),
                "caption_provider": record.get("caption_provider", ""),
                "recipe_id": record.get("recipe_id", ""),
                "split_group": record.get("split_group", ""),
                "palette_family": record.get("palette_family", ""),
                "layout_id": record.get("layout_id", ""),
            }
        )

    (split_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metadata),
        encoding="utf-8",
    )
    return metadata


def prepare_dataset(
    workbook_path: Path,
    image_dir: Path,
    synthetic_metadata: list[Path],
    output_dir: Path,
    *,
    validation_ratio: float,
    seed: int,
    zip_output: Path | None,
    require_vision_captions: bool = True,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    audit = audit_dataset(workbook_path, image_dir)
    if audit["critical_errors"]:
        raise ValueError("source dataset audit has critical errors; inspect the audit report first")

    original_records = [
        {
            "source_path": image_dir / row["image_file"],
            "text": row["effect"],
            "scene_prompt": row["scene_prompt"],
            "source": "organizer_original",
            "caption_source": "organizer_original",
            "caption_model": "organizer_provided",
            "original_file_name": row["image_file"],
            "recipe_id": "",
            "split_group": "organizer_original:scene:"
            + hashlib.sha256(
                " ".join(str(row["scene_prompt"]).lower().split()).encode("utf-8")
            ).hexdigest()[:16],
            "palette_family": "",
            "layout_id": "",
            "extra": {},
        }
        for row in load_training_rows(workbook_path)
    ]
    combined = original_records + load_synthetic_records(
        synthetic_metadata, require_vision_captions=require_vision_captions
    )
    deduplicated, removed_duplicates = _deduplicate(combined)
    train, validation = split_records(deduplicated, validation_ratio, seed)

    output_dir.mkdir(parents=True)
    train_metadata = _write_split(output_dir / "train", train)
    validation_metadata = _write_split(output_dir / "validation", validation)
    (output_dir / "source_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "source_record_count": len(combined),
        "deduplicated_record_count": len(deduplicated),
        "removed_exact_duplicates": removed_duplicates,
        "train_count": len(train_metadata),
        "validation_count": len(validation_metadata),
        "validation_ratio_requested": validation_ratio,
        "seed": seed,
        "source_counts": {
            source: sum(1 for item in deduplicated if item["source"] == source)
            for source in sorted({item["source"] for item in deduplicated})
        },
        "caption_source_counts": {
            caption_source: sum(
                1 for item in deduplicated if item["caption_source"] == caption_source
            )
            for caption_source in sorted({item["caption_source"] for item in deduplicated})
        },
        "split_strategy": "source-stratified, recipe/scene-group isolated",
        "split_group_count": len(
            {
                str(item.get("split_group") or item["content_sha256"])
                for item in deduplicated
            }
        ),
        "image_format": "RGB PNG",
        "layout": "Hugging Face ImageFolder with train and validation splits",
    }
    if zip_output:
        summary["zip_output"] = str(zip_output)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        "This directory is ready for Hugging Face datasets.load_dataset('imagefolder', data_dir=...).\n"
        "Each split contains RGB PNG images and a metadata.jsonl file.\n"
        "Use the 'text' column as the text-to-image training caption.\n",
        encoding="utf-8",
    )

    if zip_output:
        zip_output.parent.mkdir(parents=True, exist_ok=True)
        archive_base = zip_output.with_suffix("")
        produced = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_dir.parent, base_dir=output_dir.name))
        if produced != zip_output:
            produced.replace(zip_output)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a Colab-ready ImageFolder dataset.")
    parser.add_argument("--workbook", type=Path, default=Path("reference_data/训练集.xlsx"))
    parser.add_argument("--image-dir", type=Path, default=Path("reference_data/训练集图片"))
    parser.add_argument("--synthetic-metadata", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--zip-output", type=Path)
    parser.add_argument(
        "--allow-template-captions",
        action="store_true",
        help="development-only escape hatch; final training packages should use vision captions",
    )
    args = parser.parse_args(argv)
    try:
        summary = prepare_dataset(
            args.workbook,
            args.image_dir,
            args.synthetic_metadata,
            args.output_dir,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            zip_output=args.zip_output,
            require_vision_captions=not args.allow_template_captions,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
