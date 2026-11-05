"""Operational status logic separate from statistical anomaly detection."""

from __future__ import annotations

from enum import Enum
from numbers import Real
from math import isfinite

try:
    from .config import DEFAULT_CONFIG, SimulationConfig
except ImportError:  # Supports direct execution through src/main.py.
    from config import DEFAULT_CONFIG, SimulationConfig


class WarningStatus(str, Enum):
    NORMAL = "NORMAL"
    EARLY_WARNING = "EARLY WARNING"
    THRESHOLD_ALERT = "THRESHOLD ALERT"
    SENSOR_ERROR = "SENSOR ERROR"


class WarningManager:
    def __init__(self, config: SimulationConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def evaluate(
        self,
        co_ppm: float,
        co2_ppm: float,
        *,
        is_anomaly: bool = False,
        rising: bool = False,
    ) -> WarningStatus:
        """Evaluate raw sensor threshold first, then the early-warning rule."""
        if not self._valid_value(co_ppm) or not self._valid_value(co2_ppm):
            return WarningStatus.SENSOR_ERROR
        if co_ppm >= self.config.co_threshold_ppm or co2_ppm >= self.config.co2_threshold_ppm:
            return WarningStatus.THRESHOLD_ALERT
        if is_anomaly and rising:
            return WarningStatus.EARLY_WARNING
        return WarningStatus.NORMAL

    @staticmethod
    def _valid_value(value: float) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value)) and value >= 0
