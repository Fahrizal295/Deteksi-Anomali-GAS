"""Winsen ZE07-CO UART adapter.

The serial device is opened only when ``open()`` is called, so Windows dummy
mode can import this module without Raspberry Pi hardware.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from ..models import SensorReading


class ZE07COReader:
    FRAME_LENGTH = 9
    PORT = "/dev/serial0"
    BAUDRATE = 9600

    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        timeout: float = 0.25,
        read_attempts: int = 2,
        serial_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        if read_attempts < 1:
            raise ValueError("read_attempts must be positive")
        self.read_attempts = read_attempts
        self._serial_factory = serial_factory
        self._serial: Any | None = None
        self._buffer = bytearray()
        self.last_error: str | None = None

    def open(self) -> None:
        if self._serial is not None:
            return
        factory = self._serial_factory
        if factory is None:
            import serial

            factory = serial.Serial
        self._serial = factory(self.port, self.baudrate, timeout=self.timeout)

    def close(self) -> None:
        if self._serial is not None and hasattr(self._serial, "close"):
            self._serial.close()
        self._serial = None

    @staticmethod
    def validate_frame(frame: bytes) -> bool:
        if len(frame) != ZE07COReader.FRAME_LENGTH or frame[0] != 0xFF or frame[1] != 0x04:
            return False
        expected = (0xFF - (sum(frame[1:8]) & 0xFF) + 1) & 0xFF
        return frame[8] == expected

    @staticmethod
    def parse_frame(frame: bytes, monotonic_time: float | None = None) -> SensorReading:
        if not ZE07COReader.validate_frame(frame):
            raise ValueError("invalid ZE07-CO frame or checksum")
        concentration = ((frame[4] << 8) | frame[5]) * 0.1
        return SensorReading(
            timestamp=datetime.now(timezone.utc),
            monotonic_time=time.monotonic() if monotonic_time is None else monotonic_time,
            co_ppm=float(concentration),
            co2_ppm=None,
            source="SENSOR",
        )

    def read(self) -> SensorReading:
        self.open()
        for attempt in range(self.read_attempts):
            chunk = self._serial.read(self.FRAME_LENGTH)
            if chunk:
                self._buffer.extend(chunk)
            while len(self._buffer) >= self.FRAME_LENGTH:
                try:
                    start = self._buffer.index(0xFF)
                except ValueError:
                    self._buffer.clear()
                    self.last_error = "ZE07-CO UART buffer has no frame header"
                    break
                if start:
                    del self._buffer[:start]
                if len(self._buffer) < self.FRAME_LENGTH:
                    break
                frame = bytes(self._buffer[: self.FRAME_LENGTH])
                del self._buffer[: self.FRAME_LENGTH]
                if self.validate_frame(frame):
                    self.last_error = None
                    return self.parse_frame(frame)
                self.last_error = "ZE07-CO invalid frame or checksum"
            if attempt + 1 < self.read_attempts:
                continue
        self.last_error = self.last_error or "ZE07-CO UART timeout"
        raise TimeoutError(self.last_error)
