from src.sensors.stcc4_protocol import crc8_word, validate_measurement_crc


def measurement_bytes():
    values = (450, 20000, 30000, 0)
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
