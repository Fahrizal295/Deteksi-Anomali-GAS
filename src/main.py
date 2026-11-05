"""Command-line pipeline for training, inference, warning, and evaluation."""

from __future__ import annotations

try:
    from .anomaly_detector import IsolationForestDetector
    from .config import DEFAULT_CONFIG
    from .dummy_generator import generate_testing_data, generate_training_data
    from .evaluation import (
        apply_warnings,
        calculate_metrics,
        calculate_warning_metrics,
        early_warning_report,
        save_results,
    )
    from .feature_extraction import extract_features
    from .preprocessing import preprocess_samples
except ImportError:  # Supports running this file directly from VS Code.
    from anomaly_detector import IsolationForestDetector
    from config import DEFAULT_CONFIG
    from dummy_generator import generate_testing_data, generate_training_data
    from evaluation import (
        apply_warnings,
        calculate_metrics,
        calculate_warning_metrics,
        early_warning_report,
        save_results,
    )
    from feature_extraction import extract_features
    from preprocessing import preprocess_samples


def main() -> None:
    training = extract_features(preprocess_samples(generate_training_data()))
    testing = extract_features(preprocess_samples(generate_testing_data()))
    detector = IsolationForestDetector(DEFAULT_CONFIG).fit(training)
    detector.save("data/models/isolation_forest.joblib")
    inference_detector = IsolationForestDetector.load("data/models/isolation_forest.joblib")
    results = apply_warnings(inference_detector.predict(testing))
    save_results(results, "data/results/stage2_results.csv")
    metrics = calculate_metrics(results)
    warning_metrics = calculate_warning_metrics(results)
    lead_times, failures = early_warning_report(results)
    print(f"Training samples (normal only): {len(training)}")
    print(f"Testing samples: {len(testing)}")
    print(f"Model-ready training samples: {int(training['feature_ready'].sum())}")
    print(f"Isolation Forest metrics: {metrics}")
    print(f"Operational warning metrics: {warning_metrics}")
    print(f"Early warning lead times:\n{lead_times.to_string(index=False)}")
    print(f"Scenarios without early warning before threshold: {failures or 'none'}")
    print("Saved: data/models/isolation_forest.joblib, data/results/stage2_results.csv")


if __name__ == "__main__":
    main()
