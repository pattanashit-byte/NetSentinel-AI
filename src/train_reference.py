"""Shared helper: train the reference Isolation Forest on benign-only
windowed features from monday_benign.csv. Used by both the live consumer
and the offline evaluation script so they score with the same detector."""
from src.detectors import IsolationForestDetector
from src.features import make_windows
from src.ingest import load_csv, clean
from src.schema import WINDOW_FEATURE_COLUMNS


def fit_reference_detector():
    benign_df = clean(load_csv("monday_benign.csv"))
    windows = make_windows(benign_df)
    X = windows[WINDOW_FEATURE_COLUMNS].to_numpy()
    detector = IsolationForestDetector(contamination=0.01)
    detector.fit(X)
    return detector, windows
