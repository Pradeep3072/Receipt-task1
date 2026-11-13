from __future__ import annotations

import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from receipt_processor import process_pdf


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_docs"
EXPECTED_CSV = ROOT / "expected_results.csv"
FIELDS = (
    ("document_type", "document type"),
    ("category", "category"),
    ("vendor", "vendor"),
    ("date", "date"),
    ("amount", "amount"),
)


def normalise_amount(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return f"{Decimal(raw.replace(',', '')):.2f}"
    except InvalidOperation:
        return raw


def main() -> int:
    if not EXPECTED_CSV.exists():
        print(f"Evaluation could not start: missing {EXPECTED_CSV}", file=sys.stderr)
        return 2
    if not SAMPLE_DIR.exists():
        print(f"Evaluation could not start: missing {SAMPLE_DIR}", file=sys.stderr)
        return 2

    with EXPECTED_CSV.open(encoding="utf-8", newline="") as stream:
        expected_rows = list(csv.DictReader(stream))
    if not expected_rows:
        print("Evaluation could not start: expected-results file has no rows", file=sys.stderr)
        return 2

    actual_rows = []
    for expected in expected_rows:
        file_path = SAMPLE_DIR / expected["file"]
        if not file_path.exists():
            actual_rows.append({"file": expected["file"], "status": "Failed", "error": "Sample PDF is missing"})
            continue
        try:
            result = process_pdf(expected["file"], file_path.read_bytes())
            actual_rows.append(result)
        except OSError as exc:
            actual_rows.append({"file": expected["file"], "status": "Failed", "error": str(exc)})

    correct = {key: 0 for key, _ in FIELDS}
    mismatches: list[tuple[str, str, str, str]] = []
    for expected, actual in zip(expected_rows, actual_rows):
        for key, label in FIELDS:
            expected_value = str(expected.get(key, "") or "").strip()
            actual_value = str(actual.get(key, "") or "").strip()
            if key == "amount":
                expected_value = normalise_amount(expected_value)
                actual_value = normalise_amount(actual_value)
            if expected_value == actual_value:
                correct[key] += 1
            else:
                mismatches.append((expected["file"], label, expected_value, actual_value))

    total = len(expected_rows)
    print("Evaluation results")
    for key, label in FIELDS:
        print(f"Accuracy for {label}: {correct[key]}/{total} ({correct[key] / total:.1%})")
    if mismatches:
        print("Mismatches:")
        for filename, label, expected_value, actual_value in mismatches:
            print(
                f"- {filename} | {label} | expected: {expected_value or '[blank]'}"
                f" | actual: {actual_value or '[blank]'}"
            )
    else:
        print("Mismatches: none")

    processing_errors = [row for row in actual_rows if row.get("error")]
    if processing_errors:
        print("Processing errors:")
        for row in processing_errors:
            print(f"- {row['file']}: {row['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
