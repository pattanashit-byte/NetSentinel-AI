"""Kafka producer: streams a cleaned CICIDS2017 CSV as one JSON message per
flow, at a configurable rate — simulating real-time event arrival from a
pre-captured dataset (this is a replay of real, labeled network flows, not a
live capture; see README for what that does and doesn't demonstrate)."""
import argparse
import json
import time

from kafka import KafkaProducer

from src.ingest import clean, load_csv

TOPIC = "security-events"


def stream_file(filename: str, bootstrap_servers: str, rate_per_sec: float, max_records: int | None):
    df = clean(load_csv(filename))
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    delay = 1.0 / rate_per_sec if rate_per_sec > 0 else 0
    sent = 0
    for _, row in df.iterrows():
        producer.send(TOPIC, row.to_dict())
        sent += 1
        if max_records and sent >= max_records:
            break
        if delay:
            time.sleep(delay)
    producer.flush()
    print(f"Sent {sent} records from {filename} to topic '{TOPIC}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="friday_ddos.csv")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--rate", type=float, default=200.0, help="records/sec")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    stream_file(args.file, args.bootstrap_servers, args.rate, args.max_records)
