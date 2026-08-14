"""Validate that evaluation gold chunks still match the current splitter."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.document_loader import load_document
from app.services.evaluation import parse_expected_list, parse_expected_pages
from app.services.splitter import split_pages


def validate(input_path: Path | None = None) -> tuple[int, list[str]]:
    input_path = input_path or PROJECT_ROOT / "data" / "eval" / "qa_pairs.csv"
    chunk_index = {}
    for document_path in sorted((PROJECT_ROOT / "data" / "samples").glob("*.pdf")):
        chunks = split_pages(load_document(document_path))
        chunk_index[document_path.name] = {chunk["chunk_id"]: chunk for chunk in chunks}

    errors: list[str] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    for row_number, row in enumerate(rows, start=2):
        source_file = row.get("source_file", "")
        chunks = chunk_index.get(source_file, {})
        expected_pages = set(parse_expected_pages(row.get("gold_pages")))
        for chunk_id in parse_expected_list(row.get("gold_chunk_ids")):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                errors.append(f"row {row_number}: {source_file} missing gold chunk {chunk_id}")
                continue
            if expected_pages and chunk.get("page") not in expected_pages:
                errors.append(
                    f"row {row_number}: chunk {chunk_id} page={chunk.get('page')} "
                    f"is not in gold_pages={sorted(expected_pages)}"
                )

    return len(rows), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate gold chunk IDs against the current document splitter.")
    parser.add_argument("--input", type=Path, default=None, help="Evaluation CSV path.")
    args = parser.parse_args()
    row_count, errors = validate(args.input)
    print(f"Validated {row_count} evaluation rows.")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("Ground truth validation passed.")


if __name__ == "__main__":
    main()
