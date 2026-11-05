"""Shared measurement schema used by sensor, pipeline, and dashboard layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SensorReading:
    timestamp: datetime
    monotonic_time: float
    co_ppm: float | None
    co2_ppm: float | None
    temperature_c: float | None = None
    humidity_rh: float | None = None
    co2_status: int | None = None
    source: str = "SENSOR"
    valid: bool = True
    error: str | None = None
    sample_id: str | None = None
    co_valid: bool | None = None
    co_error: str | None = None
    co2_valid: bool | None = None
    co2_error: str | None = None
    co_timestamp: datetime | None = None
    co2_timestamp: datetime | None = None
    cycle_start_timestamp: datetime | None = None
    cycle_end_timestamp: datetime | None = None
    cycle_start_monotonic: float | None = None
    cycle_end_monotonic: float | None = None
    co_diagnostics: str | None = None
    co2_diagnostics: str | None = None
    co_frame_id: int | None = None
    co_age_seconds: float | None = None
    co_is_fresh: bool | None = None
    co_is_expired: bool | None = None
    co_frame_state: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
