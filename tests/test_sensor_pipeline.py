from datetime import datetime, timedelta, timezone

from src.config import DEFAULT_CONFIG
from src.models import SensorReading
from src.sensor_pipeline import SensorPipeline, train_normal_detector


def reading(index: int, co: float = 8.0, co2: float = 450.0, status: int = 0):
    return SensorReading(
        datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index),
        float(index), co, co2, 25.0, 50.0, status, "DUMMY",
    )


def test_pipeline_excludes_invalid_status_from_moving_average():
    pipeline = SensorPipeline(train_normal_detector())
    for index in range(5):
        pipeline.process(reading(index))
    before = len(pipeline.valid_samples)
    invalid = pipeline.process(reading(5, status=256))
    assert invalid["status"] == "SENSOR ERROR"
    assert len(pipeline.valid_samples) == before
    assert pipeline.rows[-1]["co_ma"] is None


def test_pipeline_keeps_environmental_values_out_of_model_features():
    pipeline = SensorPipeline(train_normal_detector())
    result = None
    for index in range(5):
        result = pipeline.process(reading(index))
    assert result["temperature_c"] == 25.0
    assert result["humidity_rh"] == 50.0
    assert set(pipeline.detector.model.n_features_in_ for _ in [0]) == {2}


def test_isolation_status_is_separate_from_safety_status():
    pipeline = SensorPipeline(train_normal_detector())
    result = None
    for index in range(8):
        result = pipeline.process(reading(index, co=0.5, co2=390.0))
    assert result["if_status"] in {"ANOMALY", "NORMAL"}
    assert result["safety_status"] == "NORMAL"
    assert result["status"] == result["safety_status"]


def test_sensor_error_sets_both_status_indicators():
    pipeline = SensorPipeline(train_normal_detector())
    result = pipeline.process(reading(0, status=256))
    assert result["if_status"] == "SENSOR ERROR"
    assert result["safety_status"] == "SENSOR ERROR"
    assert result["status"] == "SENSOR ERROR"
