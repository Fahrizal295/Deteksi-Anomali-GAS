from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.acquisition_control import AcquisitionLock
from src.acquisition_service import _write_row, run
from src.dashboard import AUTO_REFRESH_INTERVAL_SECONDS, freshness_state, load_readings
from src.models import SensorReading
from src.sensor_pipeline import SensorPipeline, train_normal_detector


def sensor_reading(index: int, *, valid: bool = True) -> SensorReading:
    return SensorReading(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index),
        monotonic_time=float(index),
        co_ppm=0.5 if valid else None,
        co2_ppm=390.0 if valid else None,
        temperature_c=25.0,
        humidity_rh=50.0,
        co2_status=0x0100,
        source="SENSOR",
        valid=valid,
        error=None if valid else "I2C error",
    )


def test_sensor_mode_pipeline_runs_without_model_and_keeps_safety_separate():
    pipeline = SensorPipeline(None, model_status="NOT_TRAINED")

    row = pipeline.process(sensor_reading(0))

    assert row["model_status"] == "NOT_TRAINED"
    assert row["if_status"] == "NOT_TRAINED"
    assert row["anomaly_score"] is None
    assert row["safety_status"] == "NORMAL"
    assert row["co2_status"] == 256


def test_sensor_mode_writes_without_model_using_fake_readers(tmp_path):
    class FakeCO:
        def start(self):
            pass

        def read(self):
            return sensor_reading(0)

        def stop(self):
            pass

    class FakeCO2:
        def open(self):
            pass

        def read(self):
            reading = sensor_reading(0)
            return SensorReading(
                reading.timestamp,
                reading.monotonic_time,
                None,
                reading.co2_ppm,
                reading.temperature_c,
                reading.humidity_rh,
                reading.co2_status,
                reading.source,
                reading.valid,
            )

        def close(self):
            pass

    output = tmp_path / "sensor_live.csv"
    run(
        "sensor",
        "normal",
        output,
        tmp_path / "missing-model.joblib",
        tmp_path / "control.json",
        max_samples=1,
        lock_path=tmp_path / "service.lock",
        sensor_readers_factory=lambda: (FakeCO(), FakeCO2()),
        sleep_fn=lambda _: None,
        platform_name="posix",
    )

    frame = load_readings(output)
    assert len(frame) == 1
    assert frame.iloc[0]["model_status"] == "NOT_TRAINED"
    assert frame.iloc[0]["if_status"] == "NOT_TRAINED"
    assert frame.iloc[0]["safety_status"] == "NORMAL"


def test_sensor_mode_pipeline_uses_compatible_model_when_available():
    pipeline = SensorPipeline(train_normal_detector(), model_status="READY")

    rows = [pipeline.process(sensor_reading(index)) for index in range(6)]

    assert rows[-1]["model_status"] == "READY"
    assert rows[-1]["feature_ready"] is True
    assert rows[-1]["anomaly_score"] is not None
    assert rows[-1]["if_status"] in {"NORMAL", "ANOMALY"}


def test_invalid_sensor_data_never_becomes_normal_safety_status():
    pipeline = SensorPipeline(None, model_status="NOT_TRAINED")

    row = pipeline.process(sensor_reading(0, valid=False))

    assert row["valid"] is False
    assert row["safety_status"] == "SENSOR ERROR"
    assert row["if_status"] == "SENSOR ERROR"
    assert row["co_ma"] is None


def test_live_csv_grows_and_dashboard_reads_new_rows(tmp_path):
    output = tmp_path / "live_sensor_readings.csv"
    pipeline = SensorPipeline(None, model_status="NOT_TRAINED")

    for index in range(2):
        _write_row(output, pipeline.process(sensor_reading(index)))

    frame = load_readings(output)
    assert len(frame) == 2
    assert frame.iloc[-1]["co2_status"] == 256
    assert frame.iloc[-1]["if_status"] == "NOT_TRAINED"


def test_live_csv_rejects_incompatible_existing_header(tmp_path):
    output = tmp_path / "old.csv"
    output.write_text("timestamp,co_ppm,co2_ppm\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible header"):
        _write_row(output, SensorPipeline(None).process(sensor_reading(0)))

    assert output.read_text(encoding="utf-8") == "timestamp,co_ppm,co2_ppm\n"


def test_dashboard_freshness_states():
    now = pd.Timestamp("2026-01-01T00:00:10Z")
    live = pd.DataFrame({"timestamp": ["2026-01-01T00:00:09Z"]})
    stale = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z"]})
    empty = pd.DataFrame()

    assert freshness_state(live, now=now, stale_after_seconds=3) == ("LIVE", 1.0)
    assert freshness_state(stale, now=now, stale_after_seconds=3) == ("STALE", 10.0)
    assert freshness_state(empty, now=now) == ("OFFLINE", None)


def test_dashboard_uses_periodic_streamlit_refresh():
    assert AUTO_REFRESH_INTERVAL_SECONDS == 2


def test_acquisition_lock_rejects_second_instance(tmp_path):
    path = tmp_path / "acquisition.lock"
    with AcquisitionLock(path):
        with pytest.raises(RuntimeError, match="already holds lock"):
            with AcquisitionLock(path):
                pass
