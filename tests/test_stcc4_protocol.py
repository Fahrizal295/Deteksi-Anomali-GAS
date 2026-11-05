import sys
from types import SimpleNamespace

sys.modules.setdefault("smbus2", SimpleNamespace(SMBus=None))

from src.sensors.stcc4_protocol import crc8_word, validate_measurement_crc
from src.sensors.vendor.DFRobot_STCC4 import DFRobot_STCC4_I2C


def measurement_bytes(status=0):
    values = (450, 20000, 30000, status)
    payload = bytearray()
    for value in values:
        payload.extend((value >> 8, value & 0xFF, crc8_word(value)))
    return bytes(payload)


def test_stcc4_measurement_crc_matches_documented_word_layout():
    raw = measurement_bytes()
    assert len(raw) == 12
    assert validate_measurement_crc(raw)


def test_stcc4_measurement_crc_rejects_corruption():
    raw = bytearray(measurement_bytes())
    raw[8] ^= 0x01
    assert not validate_measurement_crc(bytes(raw))


def test_stcc4_status_uses_big_endian_bytes_and_preserves_0x0100():
    raw = measurement_bytes(status=0x0100)
    driver = DFRobot_STCC4_I2C.__new__(DFRobot_STCC4_I2C)
    driver._read_data = lambda command, length: raw

    measurement = driver.measurement()

    assert raw[9:11] == b"\x01\x00"
    assert measurement is not None
    assert measurement[3] == 0x0100
