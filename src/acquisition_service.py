"""Standalone acquisition process for dummy mode and Raspberry Pi sensors."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from .acquisition_control import read_command
from .anomaly_detector import IsolationForestDetector
from .config import DEFAULT_CONFIG
from .dummy_generator import SCENARIOS, generate_scenario
from .models import SensorReading
from .sensor_pipeline import SensorPipeline, dummy_reading, error_reading, train_normal_detector


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
    try:
        co = co_reader.read()
        co2 = co2_reader.read()
        return SensorReading(
            timestamp=co.timestamp, monotonic_time=time.monotonic(), co_ppm=co.co_ppm,
            co2_ppm=co2.co2_ppm, temperature_c=co2.temperature_c,
            humidity_rh=co2.humidity_rh, co2_status=co2.co2_status,
            source="SENSOR", valid=True,
        )
    except Exception as exc:  # hardware errors become traceable SENSOR_ERROR rows
        return error_reading("SENSOR", f"acquisition error: {exc}")


def run(mode: str, scenario: str, output: Path, model_path: Path, control: Path,
        max_samples: int | None = None) -> None:
    if mode == "sensor" and __import__("os").name == "nt":
        raise RuntimeError("--mode sensor is Raspberry Pi only; use --mode dummy on Windows")
    detector = train_normal_detector(DEFAULT_CONFIG, model_path) if not model_path.exists() else IsolationForestDetector.load(model_path)
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
    parser.add_argument("--mode", choices=("dummy", "sensor"), default="dummy")
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    parser.add_argument("--output", type=Path, default=Path("data/live_readings.csv"))
    parser.add_argument("--model", type=Path, default=Path("data/models/isolation_forest.joblib"))
    parser.add_argument("--control", type=Path, default=Path("data/acquisition_control.json"))
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    run(args.mode, args.scenario, args.output, args.model, args.control, args.max_samples)


if __name__ == "__main__":
    main()
