"""Shared valid-reading -> preprocessing -> inference pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .anomaly_detector import IsolationForestDetector
from .config import DEFAULT_CONFIG, SimulationConfig
from .dummy_generator import GasSample, generate_training_data
from .feature_extraction import extract_features
from .models import SensorReading
from .preprocessing import preprocess_samples
from .validation import validate_reading
from .warning_manager import WarningManager, WarningStatus


def train_normal_detector(
    config: SimulationConfig = DEFAULT_CONFIG, model_path: str | Path | None = None
) -> IsolationForestDetector:
    """Fit once on a separate normal-only training set."""
    samples = generate_training_data(config=config)
    frame = extract_features(preprocess_samples(samples, config.moving_average_window))
    detector = IsolationForestDetector(config).fit(frame)
    if model_path is not None:
        detector.save(model_path)
    return detector


class SensorPipeline:
    """Stateful causal pipeline; invalid readings never enter the SMA/model."""

    def __init__(
        self,
        detector: IsolationForestDetector | None,
        config: SimulationConfig = DEFAULT_CONFIG,
        model_status: str | None = None,
    ):
        self.detector = detector
        self.config = config
        self.model_status = model_status or ("READY" if detector is not None else "MODEL_UNAVAILABLE")
        self.warning_manager = WarningManager(config)
        self.valid_samples: list[GasSample] = []
        self.rows: list[dict[str, object]] = []

    def reset(self) -> None:
        self.valid_samples.clear()
        self.rows.clear()

    def process(self, reading: SensorReading) -> dict[str, object]:
        checked = validate_reading(reading)
        row = checked.as_dict()
        row.update(
            {"co_ma": None, "co2_ma": None, "feature_ready": False,
             "rate_co": None, "rate_co2": None, "anomaly_score": None, "is_anomaly": None,
             "if_status": "SENSOR ERROR" if not checked.valid else "NOT READY",
             "model_status": self.model_status,
             "safety_status": WarningStatus.SENSOR_ERROR.value if not checked.valid else WarningStatus.NORMAL.value,
             # Backward-compatible alias used by the earlier evaluation code.
             "status": WarningStatus.SENSOR_ERROR.value if not checked.valid else WarningStatus.NORMAL.value}
        )
        if not checked.valid:
            row["error"] = checked.error
            self.rows.append(row)
            return row

        sample = GasSample(checked.timestamp, float(checked.co_ppm), float(checked.co2_ppm), "sensor")
        self.valid_samples.append(sample)
        features = extract_features(
            preprocess_samples(self.valid_samples, self.config.moving_average_window)
        )
        if self.detector is None:
            latest = features.iloc[-1]
            is_anomaly = False
            if_status = self.model_status
            anomaly_score = None
        else:
            scored = self.detector.predict(features)
            latest = scored.iloc[-1]
            is_anomaly = bool(latest["is_anomaly"])
            if_status = "ANOMALY" if is_anomaly else (
                "NORMAL" if bool(latest["feature_ready"]) else "NOT READY"
            )
            anomaly_score = latest["anomaly_score"]
        rising = bool((latest.get("rate_co", 0) > 0) or (latest.get("rate_co2", 0) > 0))
        status = self.warning_manager.evaluate(
            checked.co_ppm, checked.co2_ppm, is_anomaly=is_anomaly, rising=rising
        )
        row.update(
            {"co_ma": latest["co_ma"], "co2_ma": latest["co2_ma"],
             "feature_ready": bool(latest["feature_ready"]),
             "rate_co": latest.get("rate_co"), "rate_co2": latest.get("rate_co2"),
             "anomaly_score": anomaly_score, "is_anomaly": (is_anomaly if self.detector is not None else None),
             "if_status": if_status, "safety_status": status.value,
             "status": status.value, "error": None}
        )
        self.rows.append(row)
        return row

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def dummy_reading(sample: GasSample, index: int = 0) -> SensorReading:
    return SensorReading(
        timestamp=sample.timestamp, monotonic_time=float(index), co_ppm=sample.co_ppm,
        co2_ppm=sample.co2_ppm, temperature_c=25.0, humidity_rh=50.0,
        co2_status=0, source="DUMMY", valid=True, sample_id=str(index),
    )


def error_reading(source: str, error: str) -> SensorReading:
    return SensorReading(
        timestamp=datetime.now(timezone.utc), monotonic_time=0.0, co_ppm=None,
        co2_ppm=None, source=source, valid=False, error=error,
    )
