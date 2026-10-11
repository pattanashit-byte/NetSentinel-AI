"""Kafka consumer: buffers incoming flow records into rolling windows,
scores each window with an Isolation Forest trained on benign traffic at
startup, and persists every window's score as an Alert row.

Training the detector at startup (rather than loading a versioned model
artifact) is a demo-scale simplification, called out in the README — a real
deployment would load a model trained and validated offline, not retrain on
every consumer restart.
"""
import argparse
import json

from kafka import KafkaConsumer

from src.db import Alert, get_session, init_db
from src.detectors import IsolationForestDetector
from src.features import WINDOW_SIZE, _summarize_window
from src.ingest import clean, load_csv
from src.schema import WINDOW_FEATURE_COLUMNS
from src.train_reference import fit_reference_detector

TOPIC = "security-events"


def run_consumer(bootstrap_servers: str, max_windows: int | None = None):
    print("Training reference Isolation Forest on benign traffic (monday_benign.csv)...")
    detector, _ = fit_reference_detector()
    init_db()
    session = get_session()

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        consumer_timeout_ms=15000,
    )

    import pandas as pd

    buffer = []
    windows_scored = 0
    for message in consumer:
        buffer.append(message.value)
        if len(buffer) < WINDOW_SIZE:
            continue
        chunk = pd.DataFrame(buffer[-WINDOW_SIZE:])
        summary = _summarize_window(chunk)
        X = pd.DataFrame([summary])[WINDOW_FEATURE_COLUMNS].to_numpy()
        score = float(detector.score(X)[0])
        is_anomaly = int(detector.predict(X)[0])

        session.add(Alert(
            detector="isolation_forest_live",
            window_index=windows_scored,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            dest_ports_touched=int(summary["unique_ports_touched"]),
        ))
        session.commit()
        windows_scored += 1
        buffer = buffer[-(WINDOW_SIZE - 1):]  # slide, don't reset (matches training window overlap)

        if max_windows and windows_scored >= max_windows:
            break

    print(f"Scored {windows_scored} windows, {sum(1 for a in session.query(Alert).all() if a.is_anomaly)} flagged anomalous")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    run_consumer(args.bootstrap_servers, args.max_windows)
