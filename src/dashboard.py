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

DATA_FILE = Path(os.getenv("GAS_DATA_FILE", "data/live_readings.csv"))
CONTROL_FILE = Path(os.getenv("GAS_CONTROL_FILE", "data/acquisition_control.json"))


def load_readings(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
        if "timestamp" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return frame
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def _chart(frame: pd.DataFrame, raw: str, moving_average: str, threshold: float, title: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame[raw], name="Raw", mode="lines"))
    if moving_average in frame:
        figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame[moving_average], name="Moving Average", mode="lines"))
    figure.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    if "is_anomaly" in frame:
        anomalies = frame[frame["is_anomaly"].astype(str).str.lower() == "true"]
        figure.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies[raw], name="Anomaly", mode="markers", marker_color="orange"))
    figure.update_layout(title=title, xaxis_title="Timestamp", yaxis_title="ppm", height=360)
    return figure


def _status_color(status: str) -> str:
    return {"NORMAL": "green", "EARLY WARNING": "orange", "THRESHOLD ALERT": "red", "SENSOR ERROR": "darkred"}.get(status, "gray")


def render() -> None:
    st.set_page_config(page_title="Gas CO/CO₂ Monitoring", layout="wide")
    st.title("Monitoring Gas CO dan CO₂")
    st.caption("Dashboard hanya membaca CSV; sensor dan model dijalankan oleh acquisition service terpisah.")
    with st.sidebar:
        st.header("Kontrol acquisition")
        scenario = st.selectbox("Skenario dummy", SCENARIOS)
        if st.button("Start"):
            send_command(CONTROL_FILE, "start", scenario=scenario)
        if st.button("Pause"):
            send_command(CONTROL_FILE, "pause")
        if st.button("Reset"):
            send_command(CONTROL_FILE, "reset")
        st.info(f"Sumber file: {DATA_FILE}")

    frame = load_readings()
    if frame.empty:
        st.warning("Belum ada data. Jalankan acquisition service terlebih dahulu.")
        return
    latest = frame.iloc[-1]
    safety_status = str(latest.get("safety_status", latest.get("status", "SENSOR ERROR")))
    if_status = str(latest.get("if_status", "UNKNOWN"))
    st.markdown(f"### Safety status: <span style='color:{_status_color(safety_status)}'>{safety_status}</span>", unsafe_allow_html=True)
    st.metric("Isolation Forest", if_status)
    cols = st.columns(6)
    cols[0].metric("CO (ppm)", f"{float(latest['co_ppm']):.2f}" if pd.notna(latest.get("co_ppm")) else "-")
    cols[1].metric("CO₂ (ppm)", f"{float(latest['co2_ppm']):.1f}" if pd.notna(latest.get("co2_ppm")) else "-")
    cols[2].metric("Temperature (°C)", f"{float(latest['temperature_c']):.1f}" if pd.notna(latest.get("temperature_c")) else "-")
    cols[3].metric("Humidity (%RH)", f"{float(latest['humidity_rh']):.1f}" if pd.notna(latest.get("humidity_rh")) else "-")
    cols[4].metric("Anomaly score", f"{float(latest['anomaly_score']):.4f}" if pd.notna(latest.get("anomaly_score")) else "-")
    cols[5].metric("Sumber", str(latest.get("source", "-")))
    st.caption(f"Pembacaan terakhir: {latest.get('timestamp', '-')}")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(_chart(frame, "co_ppm", "co_ma", DEFAULT_CONFIG.co_threshold_ppm, "CO"), width="stretch")
    with right:
        st.plotly_chart(_chart(frame, "co2_ppm", "co2_ma", DEFAULT_CONFIG.co2_threshold_ppm, "CO₂"), width="stretch")
    st.subheader("Riwayat peringatan")
    history = frame[frame.get("safety_status", frame["status"]).isin(["EARLY WARNING", "THRESHOLD ALERT", "SENSOR ERROR"])]
    history_columns = [column for column in ["timestamp", "if_status", "safety_status", "source", "error"] if column in history]
    st.dataframe(history[history_columns].tail(100), width="stretch")
    st.download_button("Ekspor CSV", frame.to_csv(index=False).encode("utf-8"), "gas_readings.csv", "text/csv")


if __name__ == "__main__":
    render()
