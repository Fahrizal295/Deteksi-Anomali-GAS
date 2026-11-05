"""Standalone acquisition process for dummy mode and Raspberry Pi sensors."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

from .acquisition_control import read_command
from .anomaly_detector import IsolationForestDetector
from .config import DEFAULT_CONFIG
from .dummy_generator import SCENARIOS, generate_scenario
from .models import SensorReading
from .sensor_pipeline import SensorPipeline, dummy_reading
from .validation import validate_reading

BASELINE_FIELDS = (
    "timestamp",
    "co_ppm",
    "co2_ppm",
    "temperature_c",
    "humidity_rh",
    "co2_status",
    "valid",
    "error",
    "source",
    "monotonic_time",
    "co_valid",
    "co_error",
    "co2_valid",
    "co2_error",
)


def _write_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _sensor_reading(co_reader, co2_reader) -> SensorReading:
    co = co2 = None
    co_error = co2_error = None
    try:
        co = co_reader.read()
    except Exception as exc:
        co_error = f"ZE07-CO: {exc}"
    try:
        co2 = co2_reader.read()
    except Exception as exc:
        co2_error = f"STCC4: {exc}"

    timestamp = co.timestamp if co is not None else (
        co2.timestamp if co2 is not None else datetime.now(timezone.utc)
    )
    errors = [error for error in (co_error, co2_error, getattr(co, "error", None), getattr(co2, "error", None)) if error]
    return SensorReading(
        timestamp=timestamp,
        monotonic_time=time.monotonic(),
        co_ppm=co.co_ppm if co is not None else None,
        co2_ppm=co2.co2_ppm if co2 is not None else None,
        temperature_c=co2.temperature_c if co2 is not None else None,
        humidity_rh=co2.humidity_rh if co2 is not None else None,
        co2_status=co2.co2_status if co2 is not None else None,
        source="SENSOR",
        valid=bool(co is not None and co2 is not None and co.valid and co2.valid),
        error="; ".join(errors) or None,
        co_valid=co.valid if co is not None else False,
        co_error=co_error or (co.error if co is not None else None),
        co2_valid=co2.valid if co2 is not None else False,
        co2_error=co2_error or (co2.error if co2 is not None else None),
    )


def _open_sensor_readers():
    from .sensors.stcc4 import STCC4Reader
    from .sensors.ze07_co import ZE07COReader

    co_reader, co2_reader = ZE07COReader(), STCC4Reader()
    co_reader.open()
    try:
        co2_reader.open()
    except Exception:
        co_reader.close()
        raise
    return co_reader, co2_reader


def _write_baseline_row(path: Path, reading: SensorReading) -> None:
    """Append one baseline row and refuse an incompatible existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            header = tuple(next(csv.reader(handle), ()))
        if header != BASELINE_FIELDS:
            raise ValueError(
                f"Existing baseline CSV has incompatible header {header!r}; "
                f"expected {BASELINE_FIELDS!r}. Choose a new output path."
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_FIELDS)
        if not exists:
            writer.writeheader()
        row = reading.as_dict()
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in BASELINE_FIELDS})


def collect_baseline(
    duration_seconds: float,
    output: Path,
    *,
    co_reader=None,
    co2_reader=None,
    interval_seconds: float = DEFAULT_CONFIG.sampling_interval_seconds,
    clock=time.monotonic,
    sleep_fn=time.sleep,
) -> int:
    """Collect raw sensor baseline data without loading or training a model."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    owns_readers = co_reader is None or co2_reader is None
    if (co_reader is None) != (co2_reader is None):
        raise ValueError("co_reader and co2_reader must be provided together")
    if owns_readers:
        co_reader, co2_reader = _open_sensor_readers()

    started = clock()
    next_tick = started
    count = 0
    try:
        while clock() - started < duration_seconds:
            reading = validate_reading(_sensor_reading(co_reader, co2_reader))
            _write_baseline_row(output, reading)
            count += 1
            next_tick += interval_seconds
            sleep_fn(max(0.0, next_tick - clock()))
    except KeyboardInterrupt:
        print(f"Baseline collection interrupted safely after {count} samples.")
    finally:
        if owns_readers:
            co_reader.close()
            co2_reader.close()
    return count


def run(mode: str, scenario: str, output: Path, model_path: Path, control: Path,
        max_samples: int | None = None) -> None:
    if mode == "sensor" and __import__("os").name == "nt":
        raise RuntimeError("--mode sensor is Raspberry Pi only; use --mode dummy on Windows")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No compatible offline model at {model_path}. "
            "Review a normal baseline CSV and train with src.offline_training first."
        )
    detector = IsolationForestDetector.load(model_path, DEFAULT_CONFIG)
    pipeline = SensorPipeline(detector)
    samples = generate_scenario(scenario, max_samples or 10000, config=DEFAULT_CONFIG)
    co_reader = co2_reader = None
    if mode == "sensor":
        from .sensors.stcc4 import STCC4Reader
        from .sensors.ze07_co import ZE07COReader
        co_reader, co2_reader = ZE07COReader(), STCC4Reader()
        co_reader.open(); co2_reader.open()
    paused = False
    index = 0
    next_tick = time.monotonic()
    try:
        while max_samples is None or index < max_samples:
            command = read_command(control)
            if command:
                if command.get("command") == "pause": paused = True
                if command.get("command") == "start": paused = False
                if command.get("command") == "reset": pipeline.reset(); output.unlink(missing_ok=True)
                if command.get("scenario") in SCENARIOS:
                    scenario = command["scenario"]; samples = generate_scenario(scenario, 10000, config=DEFAULT_CONFIG)
            if paused:
                time.sleep(0.1); continue
            if mode == "dummy":
                reading = dummy_reading(samples[index], index)
            else:
                reading = _sensor_reading(co_reader, co2_reader)
            _write_row(output, pipeline.process(reading))
            index += 1
            next_tick += DEFAULT_CONFIG.sampling_interval_seconds
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        if co_reader is not None: co_reader.close()
        if co2_reader is not None: co2_reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dummy", "sensor", "collect-baseline"), default="dummy")
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    parser.add_argument("--output", type=Path, default=Path("data/live_readings.csv"))
    parser.add_argument("--model", type=Path, default=Path("data/models/isolation_forest.joblib"))
    parser.add_argument("--control", type=Path, default=Path("data/acquisition_control.json"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--duration", type=float, help="Baseline collection duration in seconds")
    args = parser.parse_args()
    if args.mode == "collect-baseline":
        if args.duration is None:
            parser.error("--duration is required with --mode collect-baseline")
        if __import__("os").name == "nt":
            raise RuntimeError("--mode collect-baseline is Raspberry Pi only")
        output = args.output if args.output != Path("data/live_readings.csv") else Path("data/baseline_sensor_normal.csv")
        collect_baseline(args.duration, output)
        return
    run(args.mode, args.scenario, args.output, args.model, args.control, args.max_samples)


if __name__ == "__main__":
    main()
