"""Read-only Streamlit dashboard for the standalone acquisition service."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .acquisition_control import send_command
    from .config import DEFAULT_CONFIG
    from .dummy_generator import SCENARIOS
except ImportError:  # Streamlit executes this file as a script.
    from acquisition_control import send_command
    from config import DEFAULT_CONFIG
    from dummy_generator import SCENARIOS

DATA_FILE = Path(os.getenv("GAS_DATA_FILE", "data/live_sensor_readings.csv"))
CONTROL_FILE = Path(os.getenv("GAS_CONTROL_FILE", "data/acquisition_control.json"))
STALE_AFTER_SECONDS = max(3.0, DEFAULT_CONFIG.sampling_interval_seconds * 3.0)
AUTO_REFRESH_INTERVAL_SECONDS = 2


def load_readings(path: Path = DATA_FILE) -> pd.DataFrame:
    """Read the acquisition CSV without initializing sensors or models."""
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
        if "timestamp" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return frame
    except (OSError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()


def freshness_state(
    frame: pd.DataFrame,
    now: pd.Timestamp | None = None,
    stale_after_seconds: float = STALE_AFTER_SECONDS,
) -> tuple[str, float | None]:
    """Return LIVE, STALE, or OFFLINE from the newest valid timestamp."""
    if frame.empty or "timestamp" not in frame:
        return "OFFLINE", None
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return "OFFLINE", None
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    age = max(0.0, (current - timestamps.iloc[-1]).total_seconds())
    return ("LIVE" if age <= stale_after_seconds else "STALE"), age


def _chart(frame: pd.DataFrame, raw: str, moving_average: str, threshold: float, title: str) -> go.Figure:
    figure = go.Figure()
    if raw in frame:
        figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame[raw], name="Raw", mode="lines"))
    if moving_average in frame:
        figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame[moving_average], name="Moving Average", mode="lines"))
    figure.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    if "is_anomaly" in frame:
        anomalies = frame[frame["is_anomaly"].astype(str).str.lower() == "true"]
        if not anomalies.empty and raw in anomalies:
            figure.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies[raw], name="Anomaly", mode="markers", marker_color="orange"))
    figure.update_layout(title=title, xaxis_title="Timestamp", yaxis_title="ppm", height=360)
    return figure


def _status_color(status: str) -> str:
    return {
        "NORMAL": "green",
        "EARLY WARNING": "orange",
        "THRESHOLD ALERT": "red",
        "SENSOR ERROR": "darkred",
        "LIVE": "green",
        "STALE": "orange",
        "OFFLINE": "darkred",
    }.get(status, "gray")


def _number(row: pd.Series, name: str, digits: int = 2) -> str:
    value = row.get(name)
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _sensor_values_available(state: str, row: pd.Series | None) -> bool:
    """Allow current sensor values only when the newest row is live and valid."""
    if row is None or state != "LIVE":
        return False
    return str(row.get("valid", "False")).strip().lower() == "true"


def _safe_number(row: pd.Series | None, name: str, available: bool, digits: int = 2) -> str:
    if not available or row is None:
        return "—"
    return _number(row, name, digits)


def _compact_model_status(status: str) -> str:
    return {
        "MODEL_UNAVAILABLE": "Unavailable",
        "NOT_TRAINED": "Not trained",
        "READY": "Ready",
    }.get(status, status)


def _render_controls() -> None:
    with st.sidebar:
        st.header("Kontrol acquisition")
        selected_mode = st.radio("Mode kontrol", ("SENSOR", "DUMMY"), index=0)
        if selected_mode == "DUMMY":
            scenario = st.selectbox("Skenario dummy", SCENARIOS)
            st.caption("Mode DUMMY hanya aktif jika acquisition service dijalankan dengan --mode dummy.")
        else:
            scenario = None
            st.caption("Monitoring sensor asli; dashboard tidak membuka UART/I²C.")
        if st.button("Start"):
            send_command(CONTROL_FILE, "start", **({"scenario": scenario} if scenario else {}))
        if st.button("Pause"):
            send_command(CONTROL_FILE, "pause")
        if st.button("Reset"):
            send_command(CONTROL_FILE, "reset")
        st.info(f"Sumber file: {DATA_FILE}")


def _render_once() -> None:
    frame = load_readings()
    state, age = freshness_state(frame)
    latest = frame.iloc[-1] if not frame.empty else None
    values_available = _sensor_values_available(state, latest)

    st.markdown(
        f"### Data link: <span style='color:{_status_color(state)}'>{state}</span>"
        + (f" ({age:.1f} s)" if age is not None else ""),
        unsafe_allow_html=True,
    )

    st.subheader("Pembacaan Sensor")
    primary = st.columns(2)
    primary[0].metric("CO", _safe_number(latest, "co_ppm", values_available, 2), "ppm")
    primary[1].metric("CO₂", _safe_number(latest, "co2_ppm", values_available, 1), "ppm")
    moving_average = st.columns(2)
    moving_average[0].caption(
        f"Moving Average CO (window 5): {_safe_number(latest, 'co_ma', values_available, 2)} ppm"
    )
    moving_average[1].caption(
        f"Moving Average CO₂ (window 5): {_safe_number(latest, 'co2_ma', values_available, 1)} ppm"
    )
    secondary = st.columns(2)
    secondary[0].metric("Temperatur", _safe_number(latest, "temperature_c", values_available, 1), "°C")
    secondary[1].metric("Kelembapan", _safe_number(latest, "humidity_rh", values_available, 1), "%RH")

    st.subheader("Status Sistem")
    if latest is None:
        safety_status = "UNAVAILABLE"
        if_status = "UNAVAILABLE"
        model_status = "UNAVAILABLE"
        sensor_quality = "NO DATA"
        source = "-"
    else:
        raw_safety_status = str(latest.get("safety_status", latest.get("status", "SENSOR ERROR")))
        raw_if_status = str(latest.get("if_status", "MODEL_UNAVAILABLE"))
        model_status = str(latest.get("model_status", raw_if_status))
        safety_status = raw_safety_status if values_available else "UNAVAILABLE"
        if_status = raw_if_status if values_available else "UNAVAILABLE"
        sensor_quality = "VALID" if values_available else ("STALE DATA" if state == "STALE" else "INVALID")
        source = str(latest.get("source", "-"))
    status_cols = st.columns(6)
    status_cols[0].metric("Freshness", state)
    status_cols[1].metric("Safety", safety_status)
    status_cols[2].metric("Isolation Forest", _compact_model_status(if_status))
    status_cols[3].metric("Model", _compact_model_status(model_status))
    status_cols[4].metric("Sensor quality", sensor_quality)
    status_cols[5].metric("Source", source)
    if latest is None:
        st.warning("Belum ada data di CSV. Jalankan acquisition service terlebih dahulu.")
    elif not values_available:
        st.warning("Nilai sensor disembunyikan karena data tidak LIVE dan valid.")
    st.caption(
        f"Pembacaan terakhir: {latest.get('timestamp', '-') if latest is not None else '-'} | "
        f"Anomaly score: {_number(latest, 'anomaly_score', 4) if latest is not None and values_available else '-'} | "
        f"STCC4 raw status: {latest.get('co2_status', '-') if latest is not None else '-'} | "
        f"ZE07 state: {latest.get('co_frame_state', '-') if latest is not None else '-'}"
    )
    if frame.empty:
        return

    left, right = st.columns(2)
    with left:
        st.plotly_chart(_chart(frame, "co_ppm", "co_ma", DEFAULT_CONFIG.co_threshold_ppm, "CO"), width="stretch")
    with right:
        st.plotly_chart(_chart(frame, "co2_ppm", "co2_ma", DEFAULT_CONFIG.co2_threshold_ppm, "CO₂"), width="stretch")
    st.subheader("Riwayat Peringatan")
    safety_series = frame["safety_status"] if "safety_status" in frame else frame.get("status", pd.Series(dtype=str))
    history = frame[safety_series.isin(["EARLY WARNING", "THRESHOLD ALERT", "SENSOR ERROR"])]
    history_columns = [column for column in ["timestamp", "if_status", "safety_status", "source", "valid", "error"] if column in history]
    st.dataframe(history[history_columns].tail(100), width="stretch")
    st.download_button("Ekspor CSV", frame.to_csv(index=False).encode("utf-8"), "gas_readings.csv", "text/csv")


def render() -> None:
    st.set_page_config(page_title="Gas CO/CO₂ Monitoring", layout="wide")
    st.title("Monitoring Gas CO dan CO₂")
    st.caption("Dashboard hanya membaca CSV; sensor dan model dijalankan oleh acquisition service terpisah.")
    _render_controls()
    if hasattr(st, "fragment"):
        @st.fragment(run_every=f"{AUTO_REFRESH_INTERVAL_SECONDS}s")
        def live_view() -> None:
            _render_once()

        live_view()
    else:
        st.warning("Streamlit versi lama tidak mendukung auto-refresh fragment; gunakan versi terbaru.")
        _render_once()


if __name__ == "__main__":
    render()
