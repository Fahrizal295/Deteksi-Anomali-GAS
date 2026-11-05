"""DFRobot STCC4 SEN0678 adapter for Raspberry Pi I2C bus 1/address 0x64."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ..models import SensorReading
from ..validation import validate_stcc4_status


class STCC4Reader:
    """Thin adapter around the vendored official DFRobot Python driver."""

    def __init__(self, address: int = 0x64, driver: Any | None = None) -> None:
        self.address = address
        self._driver = driver
        self._started = False

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
        measurement = self._driver.measurement()
        if measurement is None:
            raise TimeoutError("STCC4 measurement not available yet")
        co2_ppm, temperature_c, humidity_rh, status = measurement
        valid, error = validate_stcc4_status(int(status))
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
