import joblib
import pytest

from src.anomaly_detector import (
    IsolationForestDetector,
    ModelCompatibilityError,
)
from src.dummy_generator import generate_training_data
from src.feature_extraction import extract_features
from src.preprocessing import preprocess_samples


def fitted_detector():
    frame = extract_features(preprocess_samples(generate_training_data(80)))
    return IsolationForestDetector().fit(frame)


def test_persisted_model_round_trip_contains_schema_metadata(tmp_path):
    path = tmp_path / "model.joblib"
    detector = fitted_detector()
    detector.save(path)
    loaded = IsolationForestDetector.load(path)
    assert loaded.schema_version == "2.0"
    assert loaded.feature_names == ("co_ma", "co2_ma")
    assert loaded.model.n_features_in_ == 2


def test_legacy_model_without_metadata_is_rejected(tmp_path):
    path = tmp_path / "legacy.joblib"
    detector = fitted_detector()
    del detector.schema_version
    del detector.feature_names
    del detector.preprocessing_config
    joblib.dump(detector, path)
    with pytest.raises(ModelCompatibilityError, match="missing metadata"):
        IsolationForestDetector.load(path)


def test_model_with_wrong_feature_metadata_is_rejected(tmp_path):
    path = tmp_path / "wrong-features.joblib"
    detector = fitted_detector()
    detector.feature_names = ("co_ma", "co2_ma", "temperature_c")
    joblib.dump(detector, path)
    with pytest.raises(ModelCompatibilityError, match="Incompatible model features"):
        IsolationForestDetector.load(path)
