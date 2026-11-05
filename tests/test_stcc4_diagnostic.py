from tests.hardware.diagnose_stcc4 import count_transition, decode_measurement
from tests.test_stcc4_protocol import measurement_bytes


def test_diagnostic_decodes_status_bytes_and_all_crc_blocks():
    decoded = decode_measurement(measurement_bytes(status=0x0100))

    assert decoded["status_decimal"] == 256
    assert decoded["status_hex"] == "0x0100"
    assert decoded["status_msb"] == 0x01
    assert decoded["status_lsb"] == 0x00
    assert decoded["crc_all_valid"] is True
    assert all(decoded[key] for key in (
        "crc_co2_valid",
        "crc_temperature_valid",
        "crc_humidity_valid",
        "crc_status_valid",
    ))


def test_diagnostic_marks_corrupt_block_without_hiding_raw_response():
    raw = bytearray(measurement_bytes(status=0))
    raw[11] ^= 0x01
    decoded = decode_measurement(bytes(raw))

    assert decoded["measurement_available"] is True
    assert decoded["crc_status_valid"] is False
    assert decoded["crc_all_valid"] is False
    assert decoded["raw_response_hex"]


def test_diagnostic_counts_0000_to_0100_transitions(tmp_path):
    path = tmp_path / "stcc4.csv"
    path.write_text(
        "status_decimal\n0\n256\n256\n0\n",
        encoding="utf-8",
    )

    assert count_transition(path) == 2
