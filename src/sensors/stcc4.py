"""DFRobot STCC4 SEN0678 adapter for Raspberry Pi I2C bus 1/address 0x64."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ..models import SensorReading
from ..validation import validate_stcc4_status


class STCC4Reader:
    """Thin adapter around the vendored official DFRobot Python driver."""

    def __init__(
        self,
        address: int = 0x64,
        driver: Any | None = None,
        read_attempts: int = 2,
        retry_delay_seconds: float = 0.15,
    ) -> None:
        self.address = address
        if read_attempts < 1:
            raise ValueError("read_attempts must be positive")
        self.read_attempts = read_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._driver = driver
        self._started = False
        self.total_read_attempts = 0
        self.measurement_unavailable = 0
        self.measurements_valid = 0
        self.last_error: str | None = None

    @property
    def diagnostics(self) -> dict[str, int | str | None]:
        return {
            "read_attempts": self.total_read_attempts,
            "measurement_unavailable": self.measurement_unavailable,
            "measurements_valid": self.measurements_valid,
            "last_error": self.last_error,
        }

    def open(self) -> None:
        if self._driver is None:
            from .vendor.DFRobot_STCC4 import DFRobot_STCC4_I2C

            self._driver = DFRobot_STCC4_I2C(self.address)
        if not self._started:
            if not self._driver.start_measurement():
                raise RuntimeError("STCC4 continuous measurement could not start")
            self._started = True

    def close(self) -> None:
        if self._driver is not None and self._started:
            self._driver.stop_measurement()
        self._started = False

    def read(self) -> SensorReading:
        self.open()
        measurement = None
        for attempt in range(self.read_attempts):
            self.total_read_attempts += 1
            measurement = self._driver.measurement()
            if measurement is not None:
                break
            self.measurement_unavailable += 1
            if attempt + 1 < self.read_attempts:
                time.sleep(self.retry_delay_seconds)
        if measurement is None:
            self.last_error = "STCC4 measurement not available after retries"
            raise TimeoutError(self.last_error)
        co2_ppm, temperature_c, humidity_rh, status = measurement
        valid, error = validate_stcc4_status(int(status))
        self.measurements_valid += 1
        self.last_error = error
        return SensorReading(
            timestamp=datetime.now(timezone.utc),
            monotonic_time=time.monotonic(),
            co_ppm=None,
            co2_ppm=float(co2_ppm),
            temperature_c=float(temperature_c),
            humidity_rh=float(humidity_rh),
            co2_status=int(status),
            source="SENSOR",
            valid=valid,
            error=error,
        )
