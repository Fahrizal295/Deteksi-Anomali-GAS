import csv
from datetime import datetime, timezone

from src.acquisition_service import BASELINE_FIELDS, _sensor_reading, collect_baseline
from src.models import SensorReading
from src.validation import validate_reading


class FakeReader:
    def __init__(self, reading):
        self.reading = reading
        self.closed = False

    def read(self):
        return self.reading

    def close(self):
        self.closed = True


def sensor_readings():
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    co = SensorReading(timestamp, 1.0, 0.5, None, source="SENSOR")
    co2 = SensorReading(timestamp, 1.0, None, 391.0, 25.0, 50.0, 0, "SENSOR")
    return co, co2


def test_collect_baseline_appends_sensor_fields_without_model(tmp_path):
    co, co2 = sensor_readings()
    co_reader = FakeReader(co)
    co2_reader = FakeReader(co2)
    current = [0.0]

    def clock():
        return current[0]

    def sleep(seconds):
        current[0] += max(seconds, 1.0)

    output = tmp_path / "baseline.csv"
    count = collect_baseline(
        3.0, output, co_reader=co_reader, co2_reader=co2_reader,
        clock=clock, sleep_fn=sleep,
    )
    assert count == 3
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == BASELINE_FIELDS
    assert len(rows) == 3
    assert rows[0]["co_ppm"] == "0.5"
    assert rows[0]["co2_ppm"] == "391.0"
    assert rows[0]["temperature_c"] == "25.0"
    assert rows[0]["valid"] == "True"
    assert rows[0]["co_timestamp"]
    assert rows[0]["co2_timestamp"]
    assert float(rows[0]["cycle_end_monotonic"]) >= float(rows[0]["cycle_start_monotonic"])
    assert rows[0]["co_diagnostics"] == "{}"
    assert rows[0]["co2_diagnostics"] == "{}"


def test_collect_baseline_does_not_overwrite_existing_rows(tmp_path):
    co, co2 = sensor_readings()
    output = tmp_path / "baseline.csv"
    current = [0.0]

    def clock():
        return current[0]

    def sleep(seconds):
        current[0] += max(seconds, 1.0)

    kwargs = {"co_reader": FakeReader(co), "co2_reader": FakeReader(co2), "clock": clock, "sleep_fn": sleep}
    collect_baseline(1.0, output, **kwargs)
    current[0] = 0.0
    collect_baseline(1.0, output, **kwargs)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2


def test_collect_baseline_handles_ctrl_c_and_keeps_partial_csv(tmp_path):
    co, co2 = sensor_readings()

    class InterruptingReader(FakeReader):
        calls = 0

        def read(self):
            self.calls += 1
            if self.calls > 1:
                raise KeyboardInterrupt
            return self.reading

    output = tmp_path / "baseline.csv"
    count = collect_baseline(
        3.0, output, co_reader=InterruptingReader(co), co2_reader=FakeReader(co2),
        clock=lambda: 0.0, sleep_fn=lambda _: None,
    )
    assert count == 1
    assert output.exists()


def test_partial_sensor_failure_keeps_other_sensor_value_and_diagnostics():
    co, _ = sensor_readings()

    class FailingReader:
        def read(self):
            raise TimeoutError("measurement not ready")

    combined = validate_reading(_sensor_reading(FakeReader(co), FailingReader()))
    assert combined.co_ppm == 0.5
    assert combined.co2_ppm is None
    assert combined.co_valid is True
    assert combined.co2_valid is False
    assert "STCC4:" in (combined.co2_error or "")
    assert "invalid co2_ppm" in (combined.error or "")
