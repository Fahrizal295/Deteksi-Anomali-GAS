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
    st.markdown(
        f"### Data link: <span style='color:{_status_color(state)}'>{state}</span>"
        + (f" ({age:.1f} s)" if age is not None else ""),
        unsafe_allow_html=True,
    )

    if frame.empty:
        st.warning("Belum ada data valid di CSV. Jalankan acquisition service terlebih dahulu.")
        return

    latest = frame.iloc[-1]
    safety_status = str(latest.get("safety_status", latest.get("status", "SENSOR ERROR")))
    if_status = str(latest.get("if_status", "MODEL_UNAVAILABLE"))
    model_status = str(latest.get("model_status", if_status))
    sensor_valid = str(latest.get("valid", "False")).lower() == "true"
    st.markdown(
        f"### Safety status: <span style='color:{_status_color(safety_status)}'>{safety_status}</span>",
        unsafe_allow_html=True,
    )
    status_cols = st.columns(4)
    status_cols[0].metric("Isolation Forest", if_status)
    status_cols[1].metric("Model", model_status)
    status_cols[2].metric("Sensor quality", "VALID" if sensor_valid else "SENSOR ERROR")
    status_cols[3].metric("Source", str(latest.get("source", "-")))
    cols = st.columns(6)
    cols[0].metric("CO (ppm)", _number(latest, "co_ppm"))
    cols[1].metric("CO₂ (ppm)", _number(latest, "co2_ppm", 1))
    cols[2].metric("CO MA", _number(latest, "co_ma"))
    cols[3].metric("CO₂ MA", _number(latest, "co2_ma", 1))
    cols[4].metric("Temperature (°C)", _number(latest, "temperature_c", 1))
    cols[5].metric("Humidity (%RH)", _number(latest, "humidity_rh", 1))
    st.caption(
        f"Pembacaan terakhir: {latest.get('timestamp', '-')} | "
        f"Anomaly score: {_number(latest, 'anomaly_score', 4)} | "
        f"STCC4 raw status: {latest.get('co2_status', '-')} | "
        f"ZE07 state: {latest.get('co_frame_state', '-')}"
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(_chart(frame, "co_ppm", "co_ma", DEFAULT_CONFIG.co_threshold_ppm, "CO"), width="stretch")
    with right:
        st.plotly_chart(_chart(frame, "co2_ppm", "co2_ma", DEFAULT_CONFIG.co2_threshold_ppm, "CO₂"), width="stretch")
    st.subheader("Riwayat peringatan")
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
