from __future__ import annotations

import csv
import math
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
CALIBRATION_FIELDS = FIELDS + (("currency", "currency"),)
CALIBRATION_BIN_COUNT = 5


def normalise_amount(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return f"{Decimal(raw.replace(',', '')):.2f}"
    except InvalidOperation:
        return raw


def normalise_field(field: str, value: object) -> str:
    raw = str(value or "").strip()
    if field == "amount":
        return normalise_amount(raw)
    if field == "currency":
        return raw.upper()
    return raw


def confidence_value(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return min(1.0, max(0.0, parsed))


def calibration_metrics(samples: list[tuple[float, bool]]) -> dict[str, float | int]:
    count = len(samples)
    if not count:
        return {"count": 0, "mean_confidence": 0.0, "accuracy": 0.0, "ece": 0.0, "brier": 0.0}

    mean_confidence = sum(confidence for confidence, _ in samples) / count
    accuracy = sum(correct for _, correct in samples) / count
    brier = sum((confidence - float(correct)) ** 2 for confidence, correct in samples) / count
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(CALIBRATION_BIN_COUNT)]
    for sample in samples:
        confidence = sample[0]
        bin_index = min(int(confidence * CALIBRATION_BIN_COUNT), CALIBRATION_BIN_COUNT - 1)
        bins[bin_index].append(sample)
    ece = 0.0
    for calibration_bin in bins:
        if not calibration_bin:
            continue
        bin_confidence = sum(confidence for confidence, _ in calibration_bin) / len(calibration_bin)
        bin_accuracy = sum(correct for _, correct in calibration_bin) / len(calibration_bin)
        ece += len(calibration_bin) / count * abs(bin_confidence - bin_accuracy)
    return {
        "count": count,
        "mean_confidence": mean_confidence,
        "accuracy": accuracy,
        "ece": ece,
        "brier": brier,
    }


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
    confidence_samples: dict[str, list[tuple[float, bool]]] = {
        label: [] for _, label in CALIBRATION_FIELDS
    }
    confidence_samples["overall document"] = []
    for expected, actual in zip(expected_rows, actual_rows):
        for key, label in FIELDS:
            expected_value = normalise_field(key, expected.get(key, ""))
            actual_value = normalise_field(key, actual.get(key, ""))
            if expected_value == actual_value:
                correct[key] += 1
            else:
                mismatches.append((expected["file"], label, expected_value, actual_value))

        field_confidence = actual.get("field_confidence", {})
        if not isinstance(field_confidence, dict):
            field_confidence = {}
        for key, label in CALIBRATION_FIELDS:
            expected_value = normalise_field(key, expected.get(key, ""))
            if not expected_value:
                continue  # Blank expected values are not applicable calibration examples.
            score = confidence_value(field_confidence.get(key))
            if score is None:
                continue
            actual_value = normalise_field(key, actual.get(key, ""))
            confidence_samples[label].append((score, expected_value == actual_value))

        complete_expected = all(
            normalise_field(key, expected.get(key, ""))
            for key, _ in CALIBRATION_FIELDS
        )
        overall_score = confidence_value(actual.get("confidence"))
        if complete_expected and overall_score is not None:
            document_correct = all(
                normalise_field(key, expected.get(key, ""))
                == normalise_field(key, actual.get(key, ""))
                for key, _ in CALIBRATION_FIELDS
            )
            confidence_samples["overall document"].append((overall_score, document_correct))

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

    print("\nConfidence calibration (descriptive on these fixtures only)")
    print("Field                 N   Mean confidence   Accuracy   ECE     Brier")
    for label, samples in confidence_samples.items():
        metrics = calibration_metrics(samples)
        if not metrics["count"]:
            print(f"{label:<21} 0   n/a              n/a        n/a     n/a")
            continue
        print(
            f"{label:<21} {metrics['count']:>2}  "
            f"{metrics['mean_confidence']:.3f}            {metrics['accuracy']:.3f}    "
            f"{metrics['ece']:.3f}   {metrics['brier']:.3f}"
        )
    print(
        "Overall document confidence is evaluated only for fixtures with all six "
        "expected fields populated. Lower ECE and Brier are better; this small "
        "synthetic set is not enough to establish reliable calibration."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
