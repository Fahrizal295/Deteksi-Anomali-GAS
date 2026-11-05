"""Reproducible synthetic CO and CO2 data for development and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np

try:
    from .config import DEFAULT_CONFIG, SimulationConfig
except ImportError:  # Supports direct execution through src/main.py.
    from config import DEFAULT_CONFIG, SimulationConfig

SCENARIOS = ("normal", "gradual_increase", "sudden_spike", "noise", "threshold_crossing")


@dataclass(frozen=True)
class GasSample:
    """One timestamped gas observation with its ground-truth scenario label."""

    timestamp: datetime
    co_ppm: float
    co2_ppm: float
    scenario: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _timestamps(count: int, start_time: datetime, interval_seconds: int) -> list[datetime]:
    return [start_time + timedelta(seconds=index * interval_seconds) for index in range(count)]


def generate_scenario(
    scenario: str,
    n_samples: int = 60,
    seed: int = DEFAULT_CONFIG.random_seed,
    config: SimulationConfig = DEFAULT_CONFIG,
    start_time: datetime | None = None,
) -> list[GasSample]:
    """Generate one deterministic scenario without using testing data for training."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Expected one of {SCENARIOS}.")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    start = start_time or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    t = np.arange(n_samples, dtype=float)

    base_co = np.full(n_samples, 8.0)
    base_co2 = np.full(n_samples, 450.0)
    co_noise = rng.normal(0.0, 0.6, n_samples)
    co2_noise = rng.normal(0.0, 10.0, n_samples)

    if scenario == "normal":
        co = base_co + co_noise
        co2 = base_co2 + co2_noise
    elif scenario == "gradual_increase":
        co = base_co + np.linspace(0.0, config.co_threshold_ppm * 0.75, n_samples) + co_noise
        co2 = base_co2 + np.linspace(0.0, config.co2_threshold_ppm * 0.75, n_samples) + co2_noise
    elif scenario == "sudden_spike":
        spike_start = max(1, n_samples // 2)
        co = base_co + co_noise
        co2 = base_co2 + co2_noise
        co[spike_start:] += config.co_threshold_ppm * 1.4
        co2[spike_start:] += config.co2_threshold_ppm * 1.4
    elif scenario == "noise":
        co = base_co + rng.normal(0.0, 5.0, n_samples)
        co2 = base_co2 + rng.normal(0.0, 90.0, n_samples)
    else:  # threshold_crossing
        co = base_co + np.linspace(0.0, config.co_threshold_ppm * 1.25, n_samples) + co_noise
        co2 = base_co2 + np.linspace(0.0, config.co2_threshold_ppm * 1.25, n_samples) + co2_noise

    return [
        GasSample(timestamp, max(0.0, float(co_value)), max(0.0, float(co2_value)), scenario)
        for timestamp, co_value, co2_value in zip(
            _timestamps(n_samples, start, config.sampling_interval_seconds), co, co2
        )
    ]


def generate_training_data(
    n_samples: int = 120,
    seed: int = DEFAULT_CONFIG.random_seed,
    config: SimulationConfig = DEFAULT_CONFIG,
) -> list[GasSample]:
    """Generate normal-only data reserved for fitting future preprocessing/models."""
    return generate_scenario("normal", n_samples, seed, config)


def generate_testing_data(
    n_samples_per_scenario: int = 60,
    seed: int = DEFAULT_CONFIG.random_seed,
    config: SimulationConfig = DEFAULT_CONFIG,
) -> list[GasSample]:
    """Generate labeled testing data, intentionally separate from training data."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    samples: list[GasSample] = []
    for index, scenario in enumerate(SCENARIOS):
        scenario_start = start + timedelta(
            seconds=index * n_samples_per_scenario * config.sampling_interval_seconds
        )
        samples.extend(
            generate_scenario(scenario, n_samples_per_scenario, seed + index, config, scenario_start)
        )
    return samples


def samples_to_records(samples: Iterable[GasSample]) -> list[dict[str, object]]:
    return [sample.as_dict() for sample in samples]
