
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    from .config import DEFAULT_CONFIG, SimulationConfig
    from .feature_extraction import FEATURE_COLUMNS, model_matrix
except ImportError:  # Supports direct execution through src/main.py.
    from config import DEFAULT_CONFIG, SimulationConfig
    from feature_extraction import FEATURE_COLUMNS, model_matrix


MODEL_SCHEMA_VERSION = "2.0"


class ModelCompatibilityError(ValueError):
    """Raised when a persisted model cannot be used by the active pipeline."""


class IsolationForestDetector:
    """Fit on normal data and score later observations independently."""

    def __init__(self, config: SimulationConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=config.isolation_forest_estimators,
            contamination=config.isolation_forest_contamination,
            random_state=config.isolation_forest_random_state,
        )
        self.is_fitted = False
        self.schema_version = MODEL_SCHEMA_VERSION
        self.feature_names = tuple(FEATURE_COLUMNS)
        self.preprocessing_config = self._preprocessing_config(config)

    @staticmethod
    def _preprocessing_config(config: SimulationConfig) -> dict[str, object]:
        return {
            "moving_average_window": config.moving_average_window,
            "sampling_interval_seconds": config.sampling_interval_seconds,
        }

    def validate_compatibility(self, config: SimulationConfig = DEFAULT_CONFIG) -> None:
        expected_features = tuple(FEATURE_COLUMNS)
        expected_preprocessing = self._preprocessing_config(config)
        missing = [
            name
            for name in ("schema_version", "feature_names", "preprocessing_config")
            if not hasattr(self, name)
        ]
        if missing:
            raise ModelCompatibilityError(
                "Persisted model is missing metadata: " + ", ".join(missing) +
                ". Retrain it offline with the current schema."
            )
        if self.schema_version != MODEL_SCHEMA_VERSION:
            raise ModelCompatibilityError(
                f"Incompatible model schema version {self.schema_version!r}; "
                f"expected {MODEL_SCHEMA_VERSION!r}. Retrain offline."
            )
        if tuple(self.feature_names) != expected_features:
            raise ModelCompatibilityError(
                f"Incompatible model features {tuple(self.feature_names)!r}; "
                f"expected {expected_features!r}. Retrain offline."
            )
        if len(self.feature_names) != getattr(self.model, "n_features_in_", None):
            raise ModelCompatibilityError(
                "Model feature metadata count does not match the fitted estimator."
            )
        if dict(self.preprocessing_config) != expected_preprocessing:
            raise ModelCompatibilityError(
                f"Incompatible preprocessing config {self.preprocessing_config!r}; "
                f"expected {expected_preprocessing!r}. Retrain offline."
            )

    def fit(self, training_frame) -> "IsolationForestDetector":
        if "scenario" in training_frame and not (training_frame["scenario"] == "normal").all():
            raise ValueError("Isolation Forest training data must contain only normal samples")
        matrix = model_matrix(training_frame)
        if len(matrix) < 2:
            raise ValueError("At least two feature-ready samples are required for training")
        self.scaler.fit(matrix)
        self.model.fit(self.scaler.transform(matrix))
        self.is_fitted = True
        self.schema_version = MODEL_SCHEMA_VERSION
        self.feature_names = tuple(FEATURE_COLUMNS)
        self.preprocessing_config = self._preprocessing_config(self.config)
        self.validate_compatibility(self.config)
        return self

    def predict(self, frame):
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before inference")
        self.validate_compatibility(self.config)
        result = frame.copy()
        result["anomaly_score"] = np.nan
        result["is_anomaly"] = False
        ready = result["feature_ready"]
        matrix = result.loc[ready, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        if len(matrix):
            scaled = self.scaler.transform(matrix)
            result.loc[ready, "anomaly_score"] = self.model.decision_function(scaled)
            result.loc[ready, "is_anomaly"] = self.model.predict(scaled) == -1
        return result

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted detector")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(
        cls, path: str | Path, config: SimulationConfig = DEFAULT_CONFIG
    ) -> "IsolationForestDetector":
        detector = joblib.load(path)
        if not isinstance(detector, cls):
            raise TypeError("Saved object is not an IsolationForestDetector")
        detector.validate_compatibility(config)
        return detector
