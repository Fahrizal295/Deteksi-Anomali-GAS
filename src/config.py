"""Central configuration for the gas anomaly detection pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters shared by the simulator and preprocessing pipeline."""

    co_threshold_ppm: float = 35.0
    co2_threshold_ppm: float = 1000.0
    sampling_interval_seconds: int = 1
    moving_average_window: int = 5
    comparable_moving_average_windows: tuple[int, ...] = (3, 5, 7)
    isolation_forest_estimators: int = 100
    isolation_forest_contamination: float = 0.03
    isolation_forest_random_state: int = 42
    random_seed: int = 42


DEFAULT_CONFIG = SimulationConfig()
