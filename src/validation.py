"""Validation rules shared by hardware adapters and the acquisition service."""

from __future__ import annotations

import math
from dataclasses import replace

from .models import SensorReading

# STCC4 reports the raw status word. The acquisition policy explicitly accepts
# both observed normal statuses and preserves the raw value for audit. CRC,
# I2C/read availability, and numeric checks remain independent requirements.
STCC4_STATUS_OK = 0
STCC4_STATUS_0100 = 0x0100
STCC4_STATUS_TESTING_MODE_BYTE10 = 4
STCC4_ACCEPTED_STATUSES = frozenset({STCC4_STATUS_OK, STCC4_STATUS_0100})
STCC4_STATUS_POLICY = "accept_0x0000_and_0x0100_raw_status"


def validate_stcc4_status(status: int | None) -> tuple[bool, str | None]:
    if status is None:
        return False, "missing STCC4 status"
    if status in STCC4_ACCEPTED_STATUSES:
        return True, None
    if status & STCC4_STATUS_TESTING_MODE_BYTE10:
        return False, "STCC4 testing mode or non-zero status"
    return False, f"STCC4 non-zero status: {status}"


def validate_reading(reading: SensorReading) -> SensorReading:
    errors: list[str] = []
    if not reading.valid:
        errors.append(reading.error or "reading marked invalid")
    elif reading.error:
        errors.append(reading.error)
    for name, value in (("co_ppm", reading.co_ppm), ("co2_ppm", reading.co2_ppm)):
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            errors.append(f"invalid {name}")
    if reading.temperature_c is not None and not math.isfinite(float(reading.temperature_c)):
        errors.append("invalid temperature_c")
    if reading.humidity_rh is not None and not math.isfinite(float(reading.humidity_rh)):
        errors.append("invalid humidity_rh")
    if reading.co2_status is not None:
        valid_status, status_error = validate_stcc4_status(reading.co2_status)
        if not valid_status and status_error:
            errors.append(status_error)
    if errors:
        return replace(reading, valid=False, error="; ".join(errors))
    return replace(reading, valid=True, error=None)
