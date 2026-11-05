"""Review and train an Isolation Forest from a separately collected baseline CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .anomaly_detector import IsolationForestDetector
from .config import DEFAULT_CONFIG, SimulationConfig
from .dummy_generator import GasSample
from .feature_extraction import extract_features
from .preprocessing import preprocess_samples

REQUIRED_COLUMNS = ("timestamp", "co_ppm", "co2_ppm")


def review_baseline_csv(path: str | Path) -> dict[str, object]:
    """Return an auditable summary without fitting or changing the dataset."""
    source = Path(path)
    frame = pd.read_csv(source)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Baseline CSV is missing required columns: {missing}")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    co = pd.to_numeric(frame["co_ppm"], errors="coerce")
    co2 = pd.to_numeric(frame["co2_ppm"], errors="coerce")
    invalid = timestamps.isna() | co.isna() | co2.isna() | (co < 0) | (co2 < 0)
    if "scenario" in frame:
        invalid = invalid | frame["scenario"].fillna("normal").ne("normal")
    if "co_is_fresh" in frame:
        invalid = invalid | frame["co_is_fresh"].astype(str).str.lower().ne("true")
    if "co_is_expired" in frame:
        invalid = invalid | frame["co_is_expired"].astype(str).str.lower().eq("true")
    return {
        "path": str(source),
        "rows": int(len(frame)),
        "invalid_rows": int(invalid.sum()),
        "columns": list(frame.columns),
        "scenario_values": sorted(frame["scenario"].dropna().astype(str).unique().tolist())
        if "scenario" in frame else [],
        "first_timestamp": None if timestamps.dropna().empty else timestamps.min().isoformat(),
        "last_timestamp": None if timestamps.dropna().empty else timestamps.max().isoformat(),
        "co_min": None if co.dropna().empty else float(co.min()),
        "co_max": None if co.dropna().empty else float(co.max()),
        "co2_min": None if co2.dropna().empty else float(co2.min()),
        "co2_max": None if co2.dropna().empty else float(co2.max()),
    }


def _samples_from_baseline(path: str | Path) -> list[GasSample]:
    frame = pd.read_csv(path)
    report = review_baseline_csv(path)
    if report["invalid_rows"]:
        raise ValueError(
            f"Baseline contains {report['invalid_rows']} invalid/non-normal rows; "
            "review and clean it before training."
        )
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Baseline timestamps must be increasing and unique")
    return [
        GasSample(timestamp, float(co), float(co2), "normal")
        for timestamp, co, co2 in zip(timestamps, frame["co_ppm"], frame["co2_ppm"])
    ]


def train_from_baseline_csv(
    baseline_path: str | Path,
    model_path: str | Path,
    *,
    approved: bool = False,
    config: SimulationConfig = DEFAULT_CONFIG,
) -> IsolationForestDetector:
    """Train only after the caller explicitly approves the reviewed dataset."""
    report = review_baseline_csv(baseline_path)
    if not approved:
        raise PermissionError(
            "Dataset review required. Run with --review, inspect the report, "
            "then pass --approve to train."
        )
    samples = _samples_from_baseline(baseline_path)
    frame = extract_features(preprocess_samples(samples, config.moving_average_window))
    detector = IsolationForestDetector(config).fit(frame)
    detector.save(model_path)
    return detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    report = review_baseline_csv(args.baseline_csv)
    print(report)
    if args.review and not args.approve:
        return
    if not args.approve:
        raise SystemExit("Pass --review first, then --approve after reviewing the dataset.")
    train_from_baseline_csv(args.baseline_csv, args.model, approved=True)
    print(f"Saved compatible model: {args.model}")


if __name__ == "__main__":
    main()
