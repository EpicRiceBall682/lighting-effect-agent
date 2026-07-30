"""Backfill deterministic recipe grouping into legacy synthetic metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .prepare_dataset import infer_synthetic_grouping_fields


def backfill_grouping_metadata(source: Path, output: Path) -> dict[str, int]:
    records: list[dict[str, Any]] = []
    unresolved = 0
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number} is not valid JSONL") from exc
        grouping = infer_synthetic_grouping_fields(record)
        if not all(grouping.values()):
            unresolved += 1
        records.append({**record, **grouping})

    if unresolved:
        raise ValueError(f"{unresolved} records could not be assigned grouping metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(output)
    return {"total": len(records), "resolved": len(records), "unresolved": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill legacy synthetic grouping metadata.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = backfill_grouping_metadata(args.source, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
