import pandas as pd
import pytest

from src.offline_training import review_baseline_csv, train_from_baseline_csv


def baseline_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="s", tz="UTC"),
            "co_ppm": [0.5] * 12,
            "co2_ppm": [391.0] * 12,
            "source": ["SENSOR"] * 12,
        }
    )


def test_baseline_can_be_reviewed_before_training(tmp_path):
    path = tmp_path / "baseline.csv"
    baseline_frame().to_csv(path, index=False)
    report = review_baseline_csv(path)
    assert report["rows"] == 12
    assert report["invalid_rows"] == 0
    with pytest.raises(PermissionError, match="Dataset review required"):
        train_from_baseline_csv(path, tmp_path / "model.joblib")


def test_non_normal_scenario_is_rejected(tmp_path):
    path = tmp_path / "testing.csv"
    frame = baseline_frame()
    frame["scenario"] = "sudden_spike"
    frame.to_csv(path, index=False)
    report = review_baseline_csv(path)
    assert report["invalid_rows"] == 12
    with pytest.raises(ValueError, match="invalid/non-normal"):
        train_from_baseline_csv(path, tmp_path / "model.joblib", approved=True)
