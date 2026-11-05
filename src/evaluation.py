"""Evaluation metrics and result persistence for the simulation pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

try:
    from .config import DEFAULT_CONFIG
    from .warning_manager import WarningManager, WarningStatus
except ImportError:  # Supports direct execution through src/main.py.
    from config import DEFAULT_CONFIG
    from warning_manager import WarningManager, WarningStatus


def apply_warnings(frame: pd.DataFrame, manager: WarningManager | None = None) -> pd.DataFrame:
    manager = manager or WarningManager()
    result = frame.copy()
    rate_co = pd.to_numeric(result["rate_co"], errors="coerce").fillna(0)
    rate_co2 = pd.to_numeric(result["rate_co2"], errors="coerce").fillna(0)
    result["rising"] = (rate_co > 0) | (rate_co2 > 0)
    result["status"] = [
        manager.evaluate(co, co2, is_anomaly=anomaly, rising=rising).value
        for co, co2, anomaly, rising in zip(
            result["co_ppm"], result["co2_ppm"], result["is_anomaly"], result["rising"]
        )
    ]
    # Ground truth is an offline label: every non-normal dummy scenario is an
    # abnormal pattern. It is never passed into the detector as a feature.
    result["ground_truth_anomaly"] = result["scenario"] != "normal"
    # Isolation Forest metrics must use the model output only.
    result["predicted_anomaly"] = result["is_anomaly"].astype(bool)
    # Operational warning metrics are reported separately from model metrics.
    result["warning_detected"] = result["status"].isin(
        [WarningStatus.EARLY_WARNING.value, WarningStatus.THRESHOLD_ALERT.value]
    )
    result["safety_status"] = result["status"]
    if "feature_ready" in result:
        result["if_status"] = "NOT READY"
        ready = result["feature_ready"].astype(bool)
        result.loc[ready, "if_status"] = result.loc[ready, "is_anomaly"].map(
            {True: "ANOMALY", False: "NORMAL"}
        )
    else:
        result["if_status"] = result["is_anomaly"].map({True: "ANOMALY", False: "NORMAL"})
    return result


def _binary_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float | int | None]:
    if prediction_column == "predicted_anomaly" and "feature_ready" in frame:
        # Warm-up rows have no valid causal feature vector and are excluded
        # from model metrics. Raw threshold evaluation remains unfiltered.
        frame = frame[frame["feature_ready"]].copy()
    actual = frame["ground_truth_anomaly"].astype(bool)
    predicted = frame[prediction_column].astype(bool)
    normal = ~actual
    false_alarm_count = int((predicted & normal).sum())
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        actual, predicted, labels=[False, True]
    ).ravel()
    return {
        "evaluated_samples": int(len(frame)),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1_score": float(f1_score(actual, predicted, zero_division=0)),
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate": float(false_alarm_count / normal.sum()) if normal.sum() else 0.0,
    }


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Return Isolation Forest anomaly metrics, excluding warning policy."""
    return _binary_metrics(frame, "predicted_anomaly")


def calculate_warning_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Return operational warning metrics, separately from model metrics."""
    return _binary_metrics(frame, "warning_detected")


def early_warning_report(
    frame: pd.DataFrame, manager: WarningManager | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Report first warning/threshold for every scenario.

    Lead time is computed only for a threshold-crossing event with an EARLY
    WARNING strictly before its first threshold timestamp. Scenarios without
    threshold crossing receive a null threshold and null lead time.
    """
    manager = manager or WarningManager()
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    for scenario, group in frame.groupby("scenario", sort=False):
        threshold_rows = group[
            (group["co_ppm"] >= manager.config.co_threshold_ppm)
            | (group["co2_ppm"] >= manager.config.co2_threshold_ppm)
        ]
        warning_rows = group[group["status"] == WarningStatus.EARLY_WARNING.value]
        first_warning = warning_rows.iloc[0]["timestamp"] if not warning_rows.empty else None
        if threshold_rows.empty:
            rows.append(
                {
                    "scenario": scenario,
                    "first_early_warning": first_warning,
                    "first_threshold": None,
                    "threshold_crossing": False,
                    "lead_time_seconds": None,
                    "warning_before_threshold": None,
                }
            )
            continue
        threshold_time = threshold_rows.iloc[0]["timestamp"]
        prior_warning = warning_rows[warning_rows["timestamp"] < threshold_time]
        lead_time = None
        success = not prior_warning.empty
        if success:
            lead_time = float(
                (pd.Timestamp(threshold_time) - pd.Timestamp(prior_warning.iloc[0]["timestamp"])).total_seconds()
            )
        else:
            failures.append(str(scenario))
        rows.append(
            {
                "scenario": scenario,
                "first_early_warning": first_warning,
                "first_threshold": threshold_time,
                "threshold_crossing": True,
                "lead_time_seconds": lead_time,
                "warning_before_threshold": success,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "first_early_warning",
            "first_threshold",
            "threshold_crossing",
            "lead_time_seconds",
            "warning_before_threshold",
        ],
    ), failures


def save_results(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
