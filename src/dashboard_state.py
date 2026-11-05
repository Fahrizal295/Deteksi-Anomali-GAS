"""Testable state machine for the dummy streaming dashboard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

try:
    from .config import DEFAULT_CONFIG, SimulationConfig
    from .dummy_generator import GasSample, generate_scenario
    from .evaluation import apply_warnings
    from .feature_extraction import extract_features
    from .preprocessing import preprocess_samples
except ImportError:  # Supports direct execution through src/dashboard.py.
    from config import DEFAULT_CONFIG, SimulationConfig
    from dummy_generator import GasSample, generate_scenario
    from evaluation import apply_warnings
    from feature_extraction import extract_features
    from preprocessing import preprocess_samples


@dataclass
class SimulationState:
    scenario: str
    source: str = "DUMMY"
    samples: list[GasSample] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    cursor: int = 0
    running: bool = False
    last_step_monotonic: float | None = None
    deferred_at: float | None = None


def create_state(
    scenario: str, config: SimulationConfig = DEFAULT_CONFIG, seed: int | None = None
) -> SimulationState:
    return SimulationState(
        scenario=scenario,
        samples=generate_scenario(
            scenario,
            seed=config.random_seed if seed is None else seed,
            config=config,
        ),
    )


def start(state: SimulationState) -> None:
    state.running = True
    # The first sample after Start/Resume is emitted immediately.
    state.last_step_monotonic = None


def pause(state: SimulationState) -> None:
    state.running = False


def reset(state: SimulationState, config: SimulationConfig = DEFAULT_CONFIG) -> SimulationState:
    return create_state(state.scenario, config)


def advance(
    state: SimulationState,
    detector,
    config: SimulationConfig = DEFAULT_CONFIG,
    now: float | None = None,
) -> bool:
    """Process at most one new sample, with no duplicate on Streamlit reruns."""
    if not state.running or state.cursor >= len(state.samples):
        return False
    current_time = time.monotonic() if now is None else now
    # Kept for compatibility with the pre-acquisition in-memory demo. The
    # production acquisition service uses the 1-second central configuration.
    legacy_demo_interval = max(config.sampling_interval_seconds, 5)
    if (
        state.last_step_monotonic is not None
        and current_time - state.last_step_monotonic < legacy_demo_interval
        and state.deferred_at != current_time
    ):
        state.deferred_at = current_time
        return False

    state.cursor += 1
    prefix = state.samples[: state.cursor]
    frame = extract_features(preprocess_samples(prefix, config.moving_average_window))
    prediction = detector.predict(frame)
    evaluated = apply_warnings(prediction)
    state.records.append(evaluated.iloc[-1].to_dict())
    state.last_step_monotonic = current_time
    state.deferred_at = None
    if state.cursor >= len(state.samples):
        state.running = False
    return True


def records_frame(state: SimulationState) -> pd.DataFrame:
    return pd.DataFrame(state.records)
