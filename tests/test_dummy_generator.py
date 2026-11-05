from datetime import datetime, timezone

import pytest

from src.config import DEFAULT_CONFIG
from src.dummy_generator import SCENARIOS, generate_scenario, generate_testing_data, generate_training_data


def test_all_scenarios_have_required_fields_and_reproducible_values():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for scenario in SCENARIOS:
        first = generate_scenario(scenario, 10, 42, start_time=start)
        second = generate_scenario(scenario, 10, 42, start_time=start)
        assert first == second
        assert all(sample.scenario == scenario for sample in first)
        assert all(sample.co_ppm >= 0 and sample.co2_ppm >= 0 for sample in first)


def test_training_and_testing_are_separate():
    training = generate_training_data(20)
    testing = generate_testing_data(10)
    assert {sample.scenario for sample in training} == {"normal"}
    assert {sample.scenario for sample in testing} == set(SCENARIOS)


def test_threshold_crossing_reaches_both_simulation_thresholds():
    samples = generate_scenario("threshold_crossing", 60, 42)
    assert max(sample.co_ppm for sample in samples) > DEFAULT_CONFIG.co_threshold_ppm
    assert max(sample.co2_ppm for sample in samples) > DEFAULT_CONFIG.co2_threshold_ppm


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        generate_scenario("unknown")
