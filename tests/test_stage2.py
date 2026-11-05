from datetime import timedelta

import pandas as pd
import pytest

from src.anomaly_detector import IsolationForestDetector
from src.config import DEFAULT_CONFIG
from src.dummy_generator import GasSample, generate_scenario, generate_training_data
from src.evaluation import apply_warnings, calculate_metrics, early_warning_report
from src.feature_extraction import extract_features
from src.preprocessing import preprocess_samples
from src.warning_manager import WarningManager, WarningStatus


def _frame(samples, window_size=5):
    return extract_features(preprocess_samples(samples, window_size=window_size))


def test_feature_extraction_uses_timestamp_for_causal_rate():
    samples = [
        GasSample(pd.Timestamp("2025-01-01T00:00:00Z"), 1, 10, "normal"),
        GasSample(pd.Timestamp("2025-01-01T00:00:05Z"), 2, 20, "normal"),
        GasSample(pd.Timestamp("2025-01-01T00:00:10Z"), 4, 40, "normal"),
    ]
    result = _frame(samples, window_size=1)
    assert result.loc[2, "rate_co"] == 0.4
    assert not result.loc[0, "feature_ready"]


def test_detector_trains_on_normal_and_marks_anomalous_rise():
    training = _frame(generate_training_data(100))
    detector = IsolationForestDetector().fit(training)
    testing = detector.predict(_frame(generate_scenario("sudden_spike", 30, 99)))
    assert testing["is_anomaly"].any()


def test_detector_rejects_non_normal_training_data_to_prevent_leakage():
    training = _frame(generate_training_data(100))
    training.loc[training.index[-1], "scenario"] = "threshold_crossing"
    with pytest.raises(ValueError, match="only normal"):
        IsolationForestDetector().fit(training)


def test_warning_manager_checks_raw_threshold_before_anomaly_logic():
    manager = WarningManager()
    assert manager.evaluate(36, 500, is_anomaly=False) == WarningStatus.THRESHOLD_ALERT
    assert manager.evaluate(10, 500, is_anomaly=True, rising=True) == WarningStatus.EARLY_WARNING
    assert manager.evaluate(10, 500, is_anomaly=True, rising=False) == WarningStatus.NORMAL


def test_threshold_crossing_and_invalid_data():
    manager = WarningManager()
    crossing = generate_scenario("threshold_crossing", 60, DEFAULT_CONFIG.random_seed)
    assert any(
        manager.evaluate(sample.co_ppm, sample.co2_ppm) == WarningStatus.THRESHOLD_ALERT
        for sample in crossing
    )
    assert manager.evaluate(float("nan"), 400) == WarningStatus.SENSOR_ERROR
    assert manager.evaluate(-1, 400) == WarningStatus.SENSOR_ERROR


def test_evaluation_marks_threshold_and_early_warning_statuses():
    training = _frame(generate_training_data(100))
    detector = IsolationForestDetector().fit(training)
    result = apply_warnings(detector.predict(_frame(generate_scenario("threshold_crossing", 60, 42))))
    assert (result["status"] == WarningStatus.THRESHOLD_ALERT.value).any()


def test_anomaly_metrics_use_isolation_forest_only_not_threshold_status():
    frame = pd.DataFrame(
        {
            "scenario": ["normal", "threshold_crossing"],
            "is_anomaly": [False, False],
            "co_ppm": [10.0, 40.0],
            "co2_ppm": [400.0, 400.0],
            "rate_co": [0.0, 0.0],
            "rate_co2": [0.0, 0.0],
        }
    )
    result = apply_warnings(frame)
    assert result.loc[1, "status"] == WarningStatus.THRESHOLD_ALERT.value
    assert calculate_metrics(result)["recall"] == 0.0
    assert calculate_metrics(result)["true_positive"] == 0
    assert calculate_metrics(result)["false_negative"] == 1


def test_early_warning_report_includes_non_crossing_and_null_lead_time():
    training = _frame(generate_training_data(100))
    detector = IsolationForestDetector().fit(training)
    result = apply_warnings(detector.predict(_frame(generate_scenario("normal", 60, 42))))
    report, failures = early_warning_report(result)
    assert len(report) == 1
    assert bool(report.loc[0, "threshold_crossing"]) is False
    assert pd.isna(report.loc[0, "first_threshold"])
    assert pd.isna(report.loc[0, "lead_time_seconds"])
    assert failures == []


def test_feature_extraction_resets_delta_at_event_boundary():
    first = generate_scenario("normal", 6, 1)
    second = generate_scenario(
        "noise", 6, 2, start_time=first[-1].timestamp + timedelta(seconds=5)
    )
    frame = _frame(first + second)
    assert pd.isna(frame.loc[6, "delta_co"])


def test_lead_time_is_threshold_time_minus_first_prior_warning():
    frame = pd.DataFrame(
        {
            "scenario": ["event", "event", "event"],
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:00:05Z", "2025-01-01T00:00:10Z"]
            ),
            "co_ppm": [10.0, 20.0, 36.0],
            "co2_ppm": [400.0, 500.0, 500.0],
            "status": ["NORMAL", "EARLY WARNING", "THRESHOLD ALERT"],
        }
    )
    report, failures = early_warning_report(frame)
    assert report.loc[0, "lead_time_seconds"] == 5.0
    assert bool(report.loc[0, "warning_before_threshold"])
    assert failures == []
