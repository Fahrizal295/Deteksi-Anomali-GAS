"""Pure-protocol helpers for the STCC4 12-byte measurement response."""

from __future__ import annotations


def crc8_word(value: int) -> int:
    """STCC4 CRC-8 over one big-endian 16-bit word (poly 0x31, init 0xFF)."""
    crc = 0xFF
    for byte in ((value >> 8) & 0xFF, value & 0xFF):
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def validate_measurement_crc(raw_data: bytes) -> bool:
    """Validate the four CRC bytes in a STCC4 measurement response."""
    if len(raw_data) < 12:
        return False
    for offset in (0, 3, 6, 9):
        value = (raw_data[offset] << 8) | raw_data[offset + 1]
        if crc8_word(value) != raw_data[offset + 2]:
            return False
    return True
