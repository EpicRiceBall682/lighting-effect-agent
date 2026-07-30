"""Audit the organizer-provided image/caption dataset without modifying it."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


REQUIRED_COLUMNS = ("image_file", "scene_prompt", "effect")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_rows(workbook_path: Path, sheet_name: str = "save") -> list[dict[str, str]]:
    frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"workbook is missing columns: {', '.join(missing_columns)}")

    rows: list[dict[str, str]] = []
    for index, row in frame.iterrows():
        record = {name: "" if pd.isna(row[name]) else str(row[name]).strip() for name in REQUIRED_COLUMNS}
        if not all(record.values()):
            raise ValueError(f"workbook row {index + 2} has empty required values")
        rows.append(record)
    return rows


def audit_dataset(workbook_path: Path, image_dir: Path) -> dict[str, Any]:
    rows = load_training_rows(workbook_path)
    image_files = {
        path.name: path
        for path in image_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }

    listed_names = [row["image_file"] for row in rows]
    missing_files = sorted(set(listed_names).difference(image_files))
    unlisted_files = sorted(set(image_files).difference(listed_names))
    corrupt_files: list[str] = []
    format_mismatches: list[dict[str, str]] = []
    image_hashes: dict[str, list[str]] = defaultdict(list)
    format_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    alpha_ranges: Counter[str] = Counter()

    for name in listed_names:
        path = image_files.get(name)
        if path is None:
            continue
        try:
            image_hashes[sha256_file(path)].append(name)
            with Image.open(path) as image:
                image.load()
                actual_format = image.format or "unknown"
                format_counts[actual_format] += 1
                mode_counts[image.mode] += 1
                size_counts[f"{image.width}x{image.height}"] += 1
                expected = path.suffix.lstrip(".").upper()
                expected = "JPEG" if expected in {"JPG", "JPEG"} else expected
                if actual_format.upper() != expected:
                    format_mismatches.append(
                        {"file": name, "extension_implies": expected, "actual_format": actual_format}
                    )
                if "A" in image.getbands():
                    extrema = image.getchannel("A").getextrema()
                    alpha_ranges[f"{extrema[0]}-{extrema[1]}"] += 1
        except (OSError, UnidentifiedImageError) as exc:
            corrupt_files.append(f"{name}: {exc}")

    duplicate_groups = [names for names in image_hashes.values() if len(names) > 1]
    scene_counts = Counter(row["scene_prompt"] for row in rows)
    effect_counts = Counter(row["effect"] for row in rows)

    critical_errors = bool(missing_files or corrupt_files)
    return {
        "workbook": str(workbook_path),
        "image_dir": str(image_dir),
        "row_count": len(rows),
        "listed_image_count": len(set(listed_names)),
        "directory_image_count": len(image_files),
        "missing_files": missing_files,
        "unlisted_files": unlisted_files,
        "corrupt_files": corrupt_files,
        "format_counts": dict(format_counts),
        "mode_counts": dict(mode_counts),
        "size_counts": dict(size_counts),
        "alpha_ranges": dict(alpha_ranges),
        "format_mismatch_count": len(format_mismatches),
        "format_mismatch_examples": format_mismatches[:10],
        "exact_duplicate_image_groups": duplicate_groups,
        "duplicate_scene_row_count": sum(count - 1 for count in scene_counts.values() if count > 1),
        "duplicate_effect_row_count": sum(count - 1 for count in effect_counts.values() if count > 1),
        "critical_errors": critical_errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the organizer-provided training dataset.")
    parser.add_argument("--workbook", type=Path, default=Path("reference_data/训练集.xlsx"))
    parser.add_argument("--image-dir", type=Path, default=Path("reference_data/训练集图片"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = audit_dataset(args.workbook, args.image_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["critical_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
