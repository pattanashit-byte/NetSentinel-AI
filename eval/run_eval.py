"""End-to-end evaluation: train each detector on benign-only data (Monday),
then score held-out windows from the two labeled attack days (Friday DDoS,
Friday PortScan) and report precision/recall/F1/ROC-AUC/false-positive-rate/
inference latency for each approach — real numbers off real, labeled traffic,
not a synthetic self-test.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.detectors import (
    IsolationForestDetector,
    SequenceAutoencoderDetector,
    StatisticalDetector,
    SupervisedBaseline,
    timed_predict,
)
from src.features import make_windows, to_sequence_tensor
from src.ingest import clean, load_csv
from src.schema import FEATURE_COLUMNS, WINDOW_FEATURE_COLUMNS

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def false_positive_rate(y_true, y_pred) -> float:
    negatives = y_true == 0
    if negatives.sum() == 0:
        return 0.0
    return float((y_pred[negatives] == 1).sum() / negatives.sum())


def evaluate(y_true, y_pred, y_score, latency_ms) -> dict:
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4) if len(set(y_true)) > 1 else None,
        "false_positive_rate": round(false_positive_rate(y_true, y_pred), 4),
        "latency_ms_per_window": round(latency_ms, 4),
    }


def main():
    print("Loading and cleaning data...")
    benign = clean(load_csv("monday_benign.csv"))
    ddos = clean(load_csv("friday_ddos.csv"))
    portscan = clean(load_csv("friday_portscan.csv"))

    print("Building windowed aggregate features...")
    train_windows = make_windows(benign)
    eval_windows = make_windows(
        __import__("pandas").concat([ddos, portscan], ignore_index=True)
    )

    X_train = train_windows[WINDOW_FEATURE_COLUMNS].to_numpy()
    X_eval = eval_windows[WINDOW_FEATURE_COLUMNS].to_numpy()
    y_eval = (eval_windows["label"] == "ATTACK").astype(int).to_numpy()

    print(f"Train windows (benign only): {len(X_train)}")
    print(f"Eval windows (DDoS + PortScan days): {len(X_eval)} ({y_eval.sum()} attack-containing)")

    results = {}

    # --- Statistical baseline ---
    stat = StatisticalDetector(z_thresh=3.0).fit(X_train)
    preds, latency = timed_predict(stat, X_eval)
    results["statistical_zscore"] = evaluate(y_eval, preds, stat.score(X_eval), latency)

    # --- Isolation Forest (unsupervised) ---
    iso = IsolationForestDetector(contamination=0.05).fit(X_train)
    preds, latency = timed_predict(iso, X_eval)
    results["isolation_forest"] = evaluate(y_eval, preds, iso.score(X_eval), latency)

    # --- Sequence LSTM autoencoder (unsupervised, per-flow sequences) ---
    print("Training LSTM autoencoder on benign per-flow sequences...")
    train_seq = to_sequence_tensor(benign, FEATURE_COLUMNS)
    combined_attack_days = __import__("pandas").concat([ddos, portscan], ignore_index=True)
    eval_seq = to_sequence_tensor(combined_attack_days, FEATURE_COLUMNS)
    # window-level label for the sequence view (same windowing as eval_windows)
    y_eval_seq = y_eval[: len(eval_seq)]

    lstm = SequenceAutoencoderDetector(n_features=len(FEATURE_COLUMNS), epochs=8)
    lstm.fit(train_seq)
    start = time.perf_counter()
    preds = lstm.predict(eval_seq)
    latency = ((time.perf_counter() - start) / len(eval_seq)) * 1000
    results["lstm_autoencoder"] = evaluate(y_eval_seq, preds, lstm.score(eval_seq), latency)

    # --- Supervised baseline (upper-bound reference, trained WITH labels) ---
    sup = SupervisedBaseline().fit(X_eval, y_eval)  # trained and evaluated on same data on purpose: it's a ceiling, not a fair unsupervised comparison
    preds, latency = timed_predict(sup, X_eval)
    results["supervised_upper_bound_reference"] = evaluate(y_eval, preds, sup.score(X_eval), latency)
    results["supervised_upper_bound_reference"]["note"] = (
        "Trained WITH attack labels on the same eval set — an in-sample ceiling reference, "
        "not a fair comparison to the unsupervised detectors above."
    )

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
