"""Data validation and sliding-window moving averages."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .config import DEFAULT_CONFIG
    from .dummy_generator import GasSample, samples_to_records
except ImportError:  # Supports direct execution through src/main.py.
    from config import DEFAULT_CONFIG
    from dummy_generator import GasSample, samples_to_records


def moving_average(values: Sequence[float], window_size: int) -> list[float | None]:
    """Return an aligned sliding-window average; incomplete windows are ``None``."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if any(not np.isfinite(value) for value in values):
        raise ValueError("values must contain only finite numbers")

    result: list[float | None] = [None] * len(values)
    for index in range(window_size - 1, len(values)):
        result[index] = float(np.mean(values[index - window_size + 1 : index + 1]))
    return result


def preprocess_samples(
    samples: Iterable[GasSample], window_size: int = DEFAULT_CONFIG.moving_average_window
) -> pd.DataFrame:
    """Validate samples and append aligned CO/CO2 moving-average columns."""
    records = samples_to_records(samples)
    frame = pd.DataFrame(records, columns=["timestamp", "co_ppm", "co2_ppm", "scenario"])
    if frame.empty:
        frame["co_ma"] = pd.Series(dtype="float64")
        frame["co2_ma"] = pd.Series(dtype="float64")
        frame["window_ready"] = pd.Series(dtype="bool")
        return frame

    if frame[["co_ppm", "co2_ppm"]].isna().any().any():
        raise ValueError("gas concentrations cannot be missing")
    if (frame[["co_ppm", "co2_ppm"]] < 0).any().any():
        raise ValueError("gas concentrations cannot be negative")

    frame["co_ma"] = moving_average(frame["co_ppm"].tolist(), window_size)
    frame["co2_ma"] = moving_average(frame["co2_ppm"].tolist(), window_size)
    frame["window_ready"] = frame["co_ma"].notna() & frame["co2_ma"].notna()
    return frame
