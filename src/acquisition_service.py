"""Standalone acquisition process for dummy mode and Raspberry Pi sensors."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .acquisition_control import AcquisitionLock, read_command
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
    "co_timestamp",
    "co2_timestamp",
    "cycle_start_timestamp",
    "cycle_end_timestamp",
    "cycle_start_monotonic",
    "cycle_end_monotonic",
    "co_diagnostics",
    "co2_diagnostics",
    "co_frame_id",
    "co_age_seconds",
    "co_is_fresh",
    "co_is_expired",
    "co_frame_state",
)

LIVE_FIELDS = tuple(SensorReading.__dataclass_fields__) + (
    "co_ma",
    "co2_ma",
    "feature_ready",
    "rate_co",
    "rate_co2",
    "anomaly_score",
    "is_anomaly",
    "if_status",
    "model_status",
    "safety_status",
    "status",
)
DEFAULT_LIVE_OUTPUT = Path("data/live_sensor_readings.csv")


def _write_row(path: Path, row: dict[str, object]) -> None:
    """Append a live row only when the existing header is exactly compatible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = LIVE_FIELDS
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            header = tuple(next(csv.reader(handle), ()))
        if header != fields:
            raise ValueError(
                f"Existing live CSV has incompatible header {header!r}; "
                f"expected {fields!r}. Choose a new output path; old data was not changed."
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def _sensor_reading(co_reader, co2_reader) -> SensorReading:
    cycle_start_monotonic = time.monotonic()
    cycle_start_timestamp = datetime.now(timezone.utc)
    co = co2 = None
    co_error = co2_error = None
    co_snapshot = None
    if hasattr(co_reader, "snapshot"):
        try:
            co_snapshot = co_reader.snapshot()
            co = co_snapshot.reading
            if co_snapshot.is_expired:
                co_error = "ZE07-CO: frame expired"
            elif co is None:
                co_error = f"ZE07-CO: {co_snapshot.error or 'no frame available'}"
            elif not co.valid:
                co_error = f"ZE07-CO: {co.error or 'invalid frame'}"
            # snapshot.error is the background thread's last communication
            # error. It remains in co_diagnostics, but must not invalidate a
            # current valid frame that was received before that error.
        except Exception as exc:
            co_error = f"ZE07-CO snapshot: {exc}"
    else:
        try:
            co = co_reader.read()
        except Exception as exc:
            co_error = f"ZE07-CO: {exc}"
    try:
        co2 = co2_reader.read()
    except Exception as exc:
        co2_error = f"STCC4: {exc}"

    cycle_end_monotonic = time.monotonic()
    cycle_end_timestamp = datetime.now(timezone.utc)
    errors = [error for error in (co_error, co2_error, getattr(co, "error", None), getattr(co2, "error", None)) if error]
    co_diagnostics = json.dumps(getattr(co_reader, "diagnostics", {}), sort_keys=True)
    co2_diagnostics = json.dumps(getattr(co2_reader, "diagnostics", {}), sort_keys=True)
    return SensorReading(
        # Combined rows use one consistent cycle-end UTC timestamp. Individual
        # sensor timestamps remain available for communication diagnostics.
        timestamp=cycle_end_timestamp,
        monotonic_time=cycle_end_monotonic,
        co_ppm=co.co_ppm if co is not None else None,
        co2_ppm=co2.co2_ppm if co2 is not None else None,
        temperature_c=co2.temperature_c if co2 is not None else None,
        humidity_rh=co2.humidity_rh if co2 is not None else None,
        co2_status=co2.co2_status if co2 is not None else None,
        source="SENSOR",
        valid=bool(
            co is not None
            and co2 is not None
            and co.valid
            and co2.valid
            and not (co_snapshot is not None and co_snapshot.is_expired)
        ),
        error="; ".join(errors) or None,
        co_valid=(co.valid and not co_snapshot.is_expired) if co is not None and co_snapshot is not None else (co.valid if co is not None else False),
        co_error=co_error or (co.error if co is not None else None),
        co2_valid=co2.valid if co2 is not None else False,
        co2_error=co2_error or (co2.error if co2 is not None else None),
        co_timestamp=co.timestamp if co is not None else None,
        co2_timestamp=co2.timestamp if co2 is not None else None,
        cycle_start_timestamp=cycle_start_timestamp,
        cycle_end_timestamp=cycle_end_timestamp,
        cycle_start_monotonic=cycle_start_monotonic,
        cycle_end_monotonic=cycle_end_monotonic,
        co_diagnostics=co_diagnostics,
        co2_diagnostics=co2_diagnostics,
        co_frame_id=co_snapshot.frame_id if co_snapshot is not None else None,
        co_age_seconds=co_snapshot.age_seconds if co_snapshot is not None else None,
        co_is_fresh=co_snapshot.is_fresh if co_snapshot is not None else (co is not None),
        co_is_expired=co_snapshot.is_expired if co_snapshot is not None else False,
        co_frame_state=co_snapshot.frame_state if co_snapshot is not None else ("DIRECT" if co is not None else "NO_FRAME"),
    )


def _open_sensor_readers():
    from .sensors.stcc4 import STCC4Reader
    from .sensors.ze07_co import ZE07COBackgroundReader

    co_reader, co2_reader = ZE07COBackgroundReader(
        freshness_timeout_seconds=DEFAULT_CONFIG.ze07_freshness_timeout_seconds
    ), STCC4Reader()
    co_reader.start()
    try:
        co2_reader.open()
    except Exception:
        co_reader.stop()
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
            if hasattr(co_reader, "stop"):
                co_reader.stop()
            else:
                co_reader.close()
            co2_reader.close()
    return count


def run(mode: str, scenario: str, output: Path, model_path: Path, control: Path,
        max_samples: int | None = None, lock_path: Path = Path("data/acquisition_service.lock"),
        sensor_readers_factory=None, sleep_fn=time.sleep, platform_name: str | None = None) -> None:
    if mode == "sensor" and (platform_name or __import__("os").name) == "nt":
        raise RuntimeError("--mode sensor is Raspberry Pi only; use --mode dummy on Windows")
    detector = None
    model_status = "NOT_TRAINED"
    if model_path.exists():
        try:
            detector = IsolationForestDetector.load(model_path, DEFAULT_CONFIG)
            model_status = "READY"
        except Exception as exc:
            model_status = "MODEL_UNAVAILABLE"
            print(f"Isolation Forest disabled: {exc}")
    elif mode != "sensor":
        model_status = "NOT_TRAINED"
    pipeline = SensorPipeline(detector, model_status=model_status)
    samples = generate_scenario(scenario, max_samples or 10000, config=DEFAULT_CONFIG)
    co_reader = co2_reader = None
    paused = False
    index = 0
    next_tick = time.monotonic()
    with AcquisitionLock(lock_path):
        try:
            if mode == "sensor":
                from .sensors.stcc4 import STCC4Reader
                from .sensors.ze07_co import ZE07COBackgroundReader
                if sensor_readers_factory is None:
                    sensor_readers_factory = lambda: (
                        ZE07COBackgroundReader(
                            freshness_timeout_seconds=DEFAULT_CONFIG.ze07_freshness_timeout_seconds
                        ),
                        STCC4Reader(),
                    )
                co_reader, co2_reader = sensor_readers_factory()
                co_reader.start()
                co2_reader.open()
            while max_samples is None or index < max_samples:
                command = read_command(control)
                if command:
                    if command.get("command") == "pause": paused = True
                    if command.get("command") == "start": paused = False
                    if command.get("command") == "reset": pipeline.reset()
                    if mode == "dummy" and command.get("scenario") in SCENARIOS:
                        scenario = command["scenario"]
                        samples = generate_scenario(scenario, 10000, config=DEFAULT_CONFIG)
                if paused:
                    time.sleep(0.1)
                    continue
                if mode == "dummy":
                    reading = dummy_reading(samples[index], index)
                else:
                    reading = _sensor_reading(co_reader, co2_reader)
                try:
                    _write_row(output, pipeline.process(reading))
                except (OSError, ValueError) as exc:
                    print(f"Acquisition stopped: cannot write compatible live CSV: {exc}")
                    break
                index += 1
                next_tick += DEFAULT_CONFIG.sampling_interval_seconds
                sleep_fn(max(0.0, next_tick - time.monotonic()))
        finally:
            if co_reader is not None:
                if hasattr(co_reader, "stop"):
                    co_reader.stop()
                else:
                    co_reader.close()
            if co2_reader is not None:
                co2_reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dummy", "sensor", "collect-baseline"), default="dummy")
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    parser.add_argument("--output", type=Path, default=DEFAULT_LIVE_OUTPUT)
    parser.add_argument("--model", type=Path, default=Path("data/models/isolation_forest.joblib"))
    parser.add_argument("--control", type=Path, default=Path("data/acquisition_control.json"))
    parser.add_argument("--lock", type=Path, default=Path("data/acquisition_service.lock"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--duration", type=float, help="Baseline collection duration in seconds")
    args = parser.parse_args()
    if args.mode == "collect-baseline":
        if args.duration is None:
            parser.error("--duration is required with --mode collect-baseline")
        if __import__("os").name == "nt":
            raise RuntimeError("--mode collect-baseline is Raspberry Pi only")
        output = args.output if args.output != DEFAULT_LIVE_OUTPUT else Path("data/baseline_sensor_normal.csv")
        with AcquisitionLock(args.lock):
            collect_baseline(args.duration, output)
        return
    run(args.mode, args.scenario, args.output, args.model, args.control, args.max_samples, args.lock)


if __name__ == "__main__":
    main()
