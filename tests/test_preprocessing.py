import pytest

from src.dummy_generator import GasSample
from src.preprocessing import moving_average, preprocess_samples


def test_moving_average_is_aligned_and_uses_sliding_window():
    assert moving_average([1, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]


def test_moving_average_rejects_invalid_window_and_values():
    with pytest.raises(ValueError):
        moving_average([1, 2], 0)
    with pytest.raises(ValueError):
        moving_average([1, float("nan")], 2)


def test_preprocessing_preserves_required_fields_and_marks_warmup():
    samples = [GasSample(i, float(i), float(i * 10), "normal") for i in range(1, 6)]
    result = preprocess_samples(samples, window_size=3)
    assert list(result.columns) == [
        "timestamp", "co_ppm", "co2_ppm", "scenario", "co_ma", "co2_ma", "window_ready"
    ]
    assert result["window_ready"].tolist() == [False, False, True, True, True]
    assert result.loc[2, "co_ma"] == pytest.approx(2.0)
    assert result.loc[4, "co2_ma"] == pytest.approx(40.0)


def test_preprocessing_rejects_negative_concentrations():
    samples = [GasSample(1, -1.0, 10.0, "normal")]
    with pytest.raises(ValueError):
        preprocess_samples(samples)
