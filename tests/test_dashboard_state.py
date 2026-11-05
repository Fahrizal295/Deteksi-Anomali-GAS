from src.anomaly_detector import IsolationForestDetector
from src.config import DEFAULT_CONFIG
from src.dashboard_state import advance, create_state, pause, reset, start
from src.dummy_generator import generate_training_data
from src.feature_extraction import extract_features
from src.preprocessing import preprocess_samples


def _detector():
    training = extract_features(preprocess_samples(generate_training_data(100)))
    return IsolationForestDetector().fit(training)


def test_stream_state_advances_once_and_does_not_duplicate_on_fast_rerun():
    state = create_state("normal", seed=10)
    detector = _detector()
    start(state)
    assert advance(state, detector, now=0.0)
    assert state.cursor == 1
    assert len(state.records) == 1
    assert not advance(state, detector, now=1.0)
    assert state.cursor == 1
    assert len(state.records) == 1
    assert advance(state, detector, now=DEFAULT_CONFIG.sampling_interval_seconds)
    assert state.cursor == 2


def test_pause_freezes_state_and_reset_clears_records():
    state = create_state("noise", seed=11)
    detector = _detector()
    start(state)
    assert advance(state, detector, now=0.0)
    pause(state)
    assert not advance(state, detector, now=10.0)
    assert state.cursor == 1
    state = reset(state)
    assert state.cursor == 0
    assert state.records == []
    assert not state.running
