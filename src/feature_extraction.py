"""Causal feature extraction for Isolation Forest inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Research model features are intentionally limited to the two causal SMAs.
# Delta/rate remain diagnostic columns for warning interpretation and logging.
FEATURE_COLUMNS = ("co_ma", "co2_ma")


def extract_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal deltas and rates to preprocessed observations.

    Rates use the elapsed timestamp between the current and previous sample.
    No future values are used. The first row and moving-average warm-up rows
    are marked not ready for model fitting/inference.
    """
    required = {"timestamp", "co_ma", "co2_ma", "window_ready"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing preprocessing columns: {sorted(missing)}")
    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamps must be valid")
    elapsed = timestamps.diff().dt.total_seconds()
    if (elapsed.iloc[1:] <= 0).any():
        raise ValueError("timestamps must be strictly increasing")

    result["co_ma"] = pd.to_numeric(result["co_ma"], errors="coerce")
    result["co2_ma"] = pd.to_numeric(result["co2_ma"], errors="coerce")
    result["delta_co"] = result["co_ma"].diff()
    result["delta_co2"] = result["co2_ma"].diff()
    result["rate_co"] = result["delta_co"] / elapsed
    result["rate_co2"] = result["delta_co2"] / elapsed
    # Dummy evaluation contains independent events concatenated into one frame.
    # Do not let the final sample of one event influence the first sample of the next.
    if "scenario" in result:
        boundary = result["scenario"].ne(result["scenario"].shift())
        result.loc[boundary, ["delta_co", "delta_co2", "rate_co", "rate_co2"]] = np.nan
    # A first difference is causal but needs one previous observation.  Keep
    # the first row out of training/inference even when SMA window == 1.
    result["feature_ready"] = (
        result["window_ready"]
        & result[list(FEATURE_COLUMNS)].notna().all(axis=1)
        & result["delta_co"].notna()
        & result["delta_co2"].notna()
    )
    return result


def model_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return only model features, excluding scenario and operational labels."""
    if not set(FEATURE_COLUMNS).issubset(frame.columns):
        raise ValueError("Feature extraction must run before model_matrix")
    ready = frame["feature_ready"]
    values = frame.loc[ready, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("model features must be finite")
    return values
