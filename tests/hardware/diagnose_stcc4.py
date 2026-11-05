"""Raspberry Pi diagnostic for raw STCC4 measurement responses.

This script is intentionally separate from the production acquisition path.
It does not apply production validation, preprocessing, or model inference.
It records raw responses so status 0x0100 can be investigated without
changing the application's status policy.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sensors.stcc4_protocol import crc8_word, validate_measurement_crc

MEASUREMENT_COMMAND = 0xEC05
SELF_TEST_COMMAND = 0x278C  # Sensirion STCC4 datasheet, section 3.4.12.
MEASUREMENT_BYTES = 12
SELF_TEST_BYTES = 3
SELF_TEST_EXECUTION_SECONDS = 0.360
RETRY_DELAY_SECONDS = 0.150
SAMPLE_INTERVAL_SECONDS = 1.0

CSV_FIELDS = (
    "timestamp",
    "sample_index",
    "co2_ppm",
    "temperature_c",
    "humidity_rh",
    "status_decimal",
    "status_hex",
    "status_msb",
    "status_lsb",
    "raw_response_hex",
    "crc_co2_valid",
    "crc_temperature_valid",
    "crc_humidity_valid",
    "crc_status_valid",
    "crc_all_valid",
    "measurement_available",
    "read_error",
    "status_transition",
)


def _decode_word(raw_data: bytes, offset: int) -> int:
    return (raw_data[offset] << 8) | raw_data[offset + 1]


def crc_flags(raw_data: bytes) -> dict[str, bool]:
    """Return CRC validity for each of the four STCC4 response words."""
    if len(raw_data) < MEASUREMENT_BYTES:
        return {
            "crc_co2_valid": False,
            "crc_temperature_valid": False,
            "crc_humidity_valid": False,
            "crc_status_valid": False,
            "crc_all_valid": False,
        }
    flags = {
        "crc_co2_valid": crc8_word(_decode_word(raw_data, 0)) == raw_data[2],
        "crc_temperature_valid": crc8_word(_decode_word(raw_data, 3)) == raw_data[5],
        "crc_humidity_valid": crc8_word(_decode_word(raw_data, 6)) == raw_data[8],
        "crc_status_valid": crc8_word(_decode_word(raw_data, 9)) == raw_data[11],
    }
    flags["crc_all_valid"] = all(flags.values())
    return flags


def decode_measurement(raw_data: bytes) -> dict[str, Any]:
    """Decode one raw response while retaining the raw bytes and CRC results."""
    flags = crc_flags(raw_data)
    result: dict[str, Any] = {
        **flags,
        "raw_response_hex": raw_data.hex(" "),
        "measurement_available": len(raw_data) >= MEASUREMENT_BYTES,
        "co2_ppm": None,
        "temperature_c": None,
        "humidity_rh": None,
        "status_decimal": None,
        "status_hex": None,
        "status_msb": None,
        "status_lsb": None,
    }
    if len(raw_data) < MEASUREMENT_BYTES:
        return result

    co2_raw = _decode_word(raw_data, 0)
    temperature_raw = _decode_word(raw_data, 3)
    humidity_raw = _decode_word(raw_data, 6)
    status = _decode_word(raw_data, 9)
    result.update(
        co2_ppm=co2_raw,
        temperature_c=-45.0 + (175.0 * temperature_raw / 65535.0),
        humidity_rh=-6.0 + (125.0 * humidity_raw / 65535.0),
        status_decimal=status,
        status_hex=f"0x{status:04X}",
        status_msb=raw_data[9],
        status_lsb=raw_data[10],
    )
    return result


def _read_response(sensor: Any) -> tuple[bytes, str | None]:
    try:
        raw_data = sensor._read_data(MEASUREMENT_COMMAND, MEASUREMENT_BYTES)
    except Exception as exc:  # Hardware diagnostic must keep the loop alive.
        return b"", f"{type(exc).__name__}: {exc}"
    if raw_data is None:
        return b"", "measurement unavailable/NACK"
    raw_bytes = bytes(raw_data)
    if len(raw_bytes) < MEASUREMENT_BYTES:
        return raw_bytes, f"short response: {len(raw_bytes)} bytes"
    return raw_bytes, None


def run_self_test(sensor: Any) -> dict[str, Any]:
    """Run the documented STCC4 self-test when explicitly requested.

    The datasheet specifies command 0x278C, 360 ms execution time, and a
    three-byte response consisting of a 16-bit result plus CRC. This routine is
    never called unless the user passes --self-test.
    """
    if not sensor._write_cmd16(SELF_TEST_COMMAND):
        raise RuntimeError("STCC4 self-test command was not accepted")
    time.sleep(SELF_TEST_EXECUTION_SECONDS)
    raw_data = bytes(sensor._bus.read_i2c_block_data(sensor._device_addr, 0, SELF_TEST_BYTES))
    if len(raw_data) < SELF_TEST_BYTES:
        raise RuntimeError(f"short self-test response: {len(raw_data)} bytes")
    result = _decode_word(raw_data, 0)
    crc_valid = crc8_word(result) == raw_data[2]
    return {
        "raw_response_hex": raw_data.hex(" "),
        "result_decimal": result,
        "result_hex": f"0x{result:04X}",
        "crc_valid": crc_valid,
    }


def _close_sensor(sensor: Any, started: bool) -> None:
    try:
        if started:
            sensor.stop_measurement()
    finally:
        bus = getattr(sensor, "_bus", None)
        if bus is not None:
            bus.close()


def run_diagnostic(output: Path, samples: int = 60, run_self_test_first: bool = False) -> int:
    """Collect raw STCC4 diagnostics on Raspberry Pi."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output}")

    from src.sensors.vendor.DFRobot_STCC4 import DFRobot_STCC4_I2C

    output.parent.mkdir(parents=True, exist_ok=True)
    sensor = DFRobot_STCC4_I2C(addr=0x64)
    status_values: list[int] = []
    previous_status: int | None = None
    rows_written = 0
    started = False
    try:
        if sensor._bus is None:
            raise RuntimeError("could not open I2C bus 1")
        if run_self_test_first:
            print(f"Self-test: {run_self_test(sensor)}")

        if not sensor.start_measurement():
            raise RuntimeError("STCC4 continuous measurement could not start")
        started = True

        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            handle.flush()
            next_deadline = time.monotonic()
            for index in range(1, samples + 1):
                remaining = next_deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                raw_data, read_error = _read_response(sensor)
                if not raw_data and read_error:
                    time.sleep(RETRY_DELAY_SECONDS)
                    retry_data, retry_error = _read_response(sensor)
                    if retry_data:
                        raw_data, read_error = retry_data, None
                    else:
                        read_error = retry_error or read_error

                decoded = decode_measurement(raw_data)
                status = decoded["status_decimal"]
                transition = ""
                if status is not None:
                    status_values.append(status)
                    if previous_status is not None and status != previous_status:
                        transition = f"0x{previous_status:04X}->{status_hex(status)}"
                    previous_status = status
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sample_index": index,
                    **decoded,
                    "read_error": read_error or "",
                    "status_transition": transition,
                }
                writer.writerow(row)
                handle.flush()
                rows_written += 1
                print(
                    f"{index:02d}/{samples} {row['timestamp']} "
                    f"CO2={display_value(decoded['co2_ppm'])} ppm "
                    f"T={display_value(decoded['temperature_c'])} C "
                    f"RH={display_value(decoded['humidity_rh'])} % "
                    f"status={display_status(status)} "
                    f"MSB={display_byte(decoded['status_msb'])} "
                    f"LSB={display_byte(decoded['status_lsb'])} "
                    f"CRC={decoded['crc_all_valid']} "
                    f"raw={decoded['raw_response_hex']}"
                )
                next_deadline += SAMPLE_INTERVAL_SECONDS
    except KeyboardInterrupt:
        print(f"Interrupted safely; rows written: {rows_written}")
    finally:
        _close_sensor(sensor, started)

    counts = Counter(status_values)
    print(f"Saved diagnostic CSV: {output}")
    print(f"Status counts: {', '.join(f'{status_hex(k)}={v}' for k, v in sorted(counts.items())) or 'none'}")
    print(f"0x0000 <-> 0x0100 transitions: {count_transition(output)}")
    return 0


def status_hex(value: int) -> str:
    return f"0x{value:04X}"


def display_status(value: int | None) -> str:
    return "N/A" if value is None else f"{value} ({status_hex(value)})"


def display_byte(value: int | None) -> str:
    return "N/A" if value is None else f"0x{value:02X}"


def display_value(value: Any) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def count_transition(path: Path) -> int:
    previous: int | None = None
    transitions = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["status_decimal"]:
                continue
            current = int(row["status_decimal"])
            if {previous, current} == {0, 0x0100}:
                transitions += 1
            previous = current
    return transitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/stcc4_diagnostic.csv"))
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run documented STCC4 self-test before measurements",
    )
    args = parser.parse_args()
    return run_diagnostic(args.output, args.samples, args.self_test)


if __name__ == "__main__":
    raise SystemExit(main())
