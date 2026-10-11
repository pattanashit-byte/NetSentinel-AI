import numpy as np

from src.drift import DriftMonitor


def test_no_drift_when_same_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=(500, 3))
    current = rng.normal(0, 1, size=(200, 3))
    monitor = DriftMonitor(reference, feature_names=["a", "b", "c"], alpha=0.01)
    report = monitor.check(current)
    assert not report.drifted


def test_drift_detected_when_distribution_shifts():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=(500, 3))
    shifted = rng.normal(5, 1, size=(200, 3))
    monitor = DriftMonitor(reference, feature_names=["a", "b", "c"], alpha=0.01)
    report = monitor.check(shifted)
    assert report.drifted
    assert set(report.drifted_features) == {"a", "b", "c"}


def test_partial_drift_only_flags_shifted_feature():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=(500, 2))
    current = reference.copy()[:200]
    current[:, 1] = rng.normal(10, 1, size=200)  # only feature index 1 shifts
    monitor = DriftMonitor(reference, feature_names=["stable", "shifted"], alpha=0.01)
    report = monitor.check(current)
    assert report.drifted_features == ["shifted"]
