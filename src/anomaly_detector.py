
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

    def fit(self, training_frame) -> "IsolationForestDetector":
        if "scenario" in training_frame and not (training_frame["scenario"] == "normal").all():
            raise ValueError("Isolation Forest training data must contain only normal samples")
        matrix = model_matrix(training_frame)
        if len(matrix) < 2:
            raise ValueError("At least two feature-ready samples are required for training")
        self.scaler.fit(matrix)
        self.model.fit(self.scaler.transform(matrix))
        self.is_fitted = True
        return self

    def predict(self, frame):
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before inference")
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
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        detector = joblib.load(path)
        if not isinstance(detector, cls):
            raise TypeError("Saved object is not an IsolationForestDetector")
        return detector
