"""Concrete demonstration of the drift monitor: compares the windowed
feature distribution of the benign training day (Monday) against each
labeled attack day, to show the monitor correctly flags the attack days as
"drifted" relative to normal behavior — which is exactly the failure mode a
retraining trigger should catch.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.drift import DriftMonitor
from src.features import make_windows
from src.ingest import clean, load_csv
from src.schema import WINDOW_FEATURE_COLUMNS

RESULTS_PATH = Path(__file__).resolve().parent / "drift_results.json"


def main():
    benign = clean(load_csv("monday_benign.csv"))
    ddos = clean(load_csv("friday_ddos.csv"))
    portscan = clean(load_csv("friday_portscan.csv"))

    reference_windows = make_windows(benign)[WINDOW_FEATURE_COLUMNS].to_numpy()
    monitor = DriftMonitor(reference_windows, feature_names=WINDOW_FEATURE_COLUMNS, alpha=0.01)

    results = {}
    for name, df in [("friday_ddos", ddos), ("friday_portscan", portscan)]:
        windows = make_windows(df)[WINDOW_FEATURE_COLUMNS].to_numpy()
        report = monitor.check(windows)
        results[name] = {
            "drifted": report.drifted,
            "drifted_features": report.drifted_features,
            "feature_pvalues": report.feature_pvalues,
        }

    # Specificity control: a RANDOM subsample of the same reference population
    # should show little/no drift (this isolates false-positive rate from any
    # real temporal structure in the data).
    rng = np.random.default_rng(42)
    idx = rng.choice(len(reference_windows), size=len(reference_windows) // 2, replace=False)
    random_control_report = monitor.check(reference_windows[idx])
    results["benign_random_subsample_control"] = {
        "drifted": random_control_report.drifted,
        "drifted_features": random_control_report.drifted_features,
    }

    # Separate, disclosed finding: an ORDER-based split (first half vs second
    # half, i.e. earlier vs later in the day) DOES show drift — see README
    # "Known limitations" for what this means for a static reference baseline.
    half = len(reference_windows) // 2
    temporal_report = monitor.check(reference_windows[half:])
    results["benign_first_half_vs_second_half_temporal_check"] = {
        "drifted": temporal_report.drifted,
        "drifted_features": temporal_report.drifted_features,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
