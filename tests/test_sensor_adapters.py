from datetime import datetime, timezone

import pytest

from src.models import SensorReading
from src.sensors.stcc4 import STCC4Reader
from src.sensors.ze07_co import ZE07COReader
from src.validation import validate_reading, validate_stcc4_status


def ze07_frame(co_ppm: float) -> bytes:
    raw = int(co_ppm * 10)
    frame = bytearray([0xFF, 0x04, 0, 0, (raw >> 8) & 0xFF, raw & 0xFF, 0, 0, 0])
    frame[8] = (0xFF - (sum(frame[1:8]) & 0xFF) + 1) & 0xFF
    return bytes(frame)


def test_ze07_parser_validates_uart_frame():
    reading = ZE07COReader.parse_frame(ze07_frame(12.3))
    assert reading.co_ppm == 12.3
    assert ZE07COReader.validate_frame(ze07_frame(12.3))
    assert not ZE07COReader.validate_frame(ze07_frame(12.3)[:-1] + b"\x00")


def test_ze07_retries_a_transient_uart_timeout():
    class Serial:
        def __init__(self):
            self.calls = 0

        def read(self, _size):
            self.calls += 1
            return b"" if self.calls == 1 else ze07_frame(0.5)

        def close(self):
            pass

    serial = Serial()
    reading = ZE07COReader(serial_factory=lambda *args, **kwargs: serial).read()
    assert reading.co_ppm == 0.5
    assert serial.calls == 2


def test_ze07_reassembles_partial_frame_across_reads():
    frame = ze07_frame(1.2)

    class Serial:
        def __init__(self):
            self.chunks = [frame[:4], frame[4:]]

        def read(self, _size):
            return self.chunks.pop(0) if self.chunks else b""

        def close(self):
            pass

    serial = Serial()
    reader = ZE07COReader(serial_factory=lambda *args, **kwargs: serial, read_attempts=2)
    reading = reader.read()
    assert reading.co_ppm == pytest.approx(1.2)
    assert reader._buffer == bytearray()
    assert reader.diagnostics["bytes_received"] == 9


def test_ze07_reports_checksum_error_and_timeout():
    frame = bytearray(ze07_frame(1.2))
    frame[-1] ^= 0x01

    class Serial:
        def read(self, _size):
            return bytes(frame)

        def close(self):
            pass

    reader = ZE07COReader(serial_factory=lambda *args, **kwargs: Serial(), read_attempts=1)
    try:
        reader.read()
    except TimeoutError:
        pass
    else:
        raise AssertionError("corrupt frame should time out")
    assert reader.diagnostics["checksum_errors"] == 1
    assert reader.diagnostics["timeouts"] == 1


class FakeSTCC4:
    def start_measurement(self):
        return True

    def stop_measurement(self):
        return True

    def measurement(self):
        return (450, 25.0, 50.0, 0)


def test_stcc4_adapter_keeps_temperature_humidity_and_valid_status():
    reading = STCC4Reader(driver=FakeSTCC4()).read()
    assert reading.co2_ppm == 450.0
    assert reading.temperature_c == 25.0
    assert reading.humidity_rh == 50.0
    assert reading.valid is True


def test_stcc4_retries_when_measurement_is_not_ready():
    class FlakySTCC4(FakeSTCC4):
        def __init__(self):
            self.calls = 0

        def measurement(self):
            self.calls += 1
            return None if self.calls == 1 else (450, 25.0, 50.0, 0)

    driver = FlakySTCC4()
    reading = STCC4Reader(driver=driver, retry_delay_seconds=0).read()
    assert reading.valid is True
    assert driver.calls == 2


def test_status_256_is_not_silently_valid():
    assert validate_stcc4_status(256)[0] is False
    reading = SensorReading(datetime.now(timezone.utc), 0.0, 8.0, 450.0, co2_status=256)
    assert validate_reading(reading).valid is False
